import torch
import torch.nn.functional as F
import numpy as np
from api import *
import tiktoken
import json
from transformers import GPT2Tokenizer
from tools import load_jsonl_file, datetime2str, save_json_file, save_file

LOCAL_CHAT_LOGGER = None
def set_chat_logger(one):
    global LOCAL_CHAT_LOGGER
    LOCAL_CHAT_LOGGER = one


def get_tokenizer_func(model_name):
    # todo: add bloom, alpaca, llama tokenizer
    if model_name in ['gpt-3.5-turbo', 'text-davinci-003']:
        tokenizer = tiktoken.encoding_for_model(model_name)
        return tokenizer.encode
    else:
        # default: gpt2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        return tokenizer.tokenize


class Turn(object):

    def __init__(self, user_input, system_response, user_sys_text, summ, embedding):
        self.user_input = user_input
        self.system_response = system_response
        self.user_sys_text = user_sys_text
        self.summ = summ
        self.embedding = embedding
        self.content_tokens_length = 0
        self.summary_tokens_length = 0
    
    def to_json(self):
        js = {
            'user': self.user_input,
            'system': self.system_response
            # 'summary': self.summ
        }
        return js
    
    def to_json_str(self):
        js = self.to_json()
        js_str = json.dumps(js, ensure_ascii=False) + '\n'
        return js_str
    
    def to_plain_text(self):
        text = f'[User]:\n{self.user_input}\n\n'
        text += f'[System]:\n{self.system_response}\n'
        text += ('-' * 30 + '\n\n')
        return text


class ChatBot(object):

    def __init__(self, model_name) -> None:
        assert model_name in MODEL_LIST, f'model name "{model_name}" is not in {MODEL_LIST}'
        self.model_name = model_name
        self.api_func = MODEL_MAP[model_name]
        self.turbo_func = MODEL_MAP['gpt-3.5-turbo']
        self.embedding_func = MODEL_EMBEDDING_MAP[model_name]
        self.tokenize_func = get_tokenizer_func(self.model_name)
        self.history: list[Turn] = []

    def clear_history(self):
        self.history = []
    
    def roll_back(self):
        self.history.pop()
    
    def export_history(self):
        hist_lst = [one.to_json() for one in self.history]
        hist_txt_lst = [one.to_plain_text() for one in self.history]
        stamp = datetime2str()
        json_filename = f'history/{hist_lst[0]["user"][:10]}-{self.model_name}-{stamp}.json'
        txt_filename = f'history/{hist_lst[0]["user"][:10]}-{self.model_name}-{stamp}.txt'
        save_json_file(json_filename, hist_lst)
        save_file(txt_filename, hist_txt_lst)
    
    # def load_history(self, hist_file):
    #     diag_hist = load_jsonl_file(hist_file)
    #     emb_hist = load_jsonl_file(hist_file + '.emb.json')
    #     for dig, e in zip(diag_hist, emb_hist):
    #         js = {}
    #         js['text'] = dig['text']
    #         js['summ'] = dig['summ']
    #         js['embedding'] = e
    #         one = Turn(**js)
    #         self.history.append(one)
    #     self.show_history()
    
    # def show_history(self):
    #     print('\n\n-------------【history】-------------\n\n')
    #     for i, turn in enumerate(self.history):
    #         print(f'{turn.text.strip()}\n\n')
    #         # print(f'对话摘要: \n{turn.summ}\n')

    def ask(self, prompt) -> str:
        output = self.api_func(prompt)
        return output

    def is_history_need(self, prompt) -> str:
        output = self.turbo_func(prompt)
        LOCAL_CHAT_LOGGER.info(f'\n--------------\nprompt: \n{prompt}\n\n')
        LOCAL_CHAT_LOGGER.info(f'output: {output}\n--------------\n')
        if ('B' in output) or ('否' in output):
            return False
        return True

    def vectorize(self, text) -> list:
        output = self.embedding_func(text)
        return output
    
    def add_turn_history(self, turn: Turn):
        turn.content_tokens_length = len(self.tokenize_func(turn.user_sys_text))
        turn.summary_tokens_length = len(self.tokenize_func(turn.summ))
        self.history.append(turn)
    
    def get_turn_for_previous(self):
        turn = self.history[-1]
        if turn.content_tokens_length < 500:
            return turn.user_sys_text
        else:
            return turn.summ
    
    def _is_concat_history_too_long(self, index_list):
            turn_length_lst = [self.history[idx].content_tokens_length for idx in index_list]
            total_tokens = sum(turn_length_lst)
            if total_tokens > 1500:
                return True
            else:
                return False


    # todo: 检索这块需要优化，不一定非得topk，最多topk，没有也可以不加
    def get_related_turn(self, query, k=3):
        q_embedding = self.vectorize(query)
        # 只检索 [0, 上一轮)   上一轮的文本直接拼接进入对话，无需检索
        sim_lst = [
            self._similarity(q_embedding, v.embedding)
            for v in self.history[:-1]
        ]

        # convert to numpy array
        arr = np.array(sim_lst)

        # get indices and values of the top k maximum values
        topk_indices = arr.argsort()[-k:]
        topk_values = arr[topk_indices]

        # print the results
        # print(f"Top {k} indices: ", topk_indices)
        # print(f"Top {k} values: ", topk_values)

        index_value_lst = [(idx, v) for idx, v in zip(topk_indices, topk_values)]
        # print(index_value_lst)
        sorted_index_value_lst = sorted(index_value_lst, key=lambda x: x[0])
        LOCAL_CHAT_LOGGER.info(f'\n--------------\n')
        LOCAL_CHAT_LOGGER.info(f"\nTop{k}相似历史索引及其相似度: \n\n{sorted_index_value_lst}\n")
        LOCAL_CHAT_LOGGER.info(f'\n--------------\n')

        shorten_history = self._is_concat_history_too_long(topk_indices)

        retrieve_history_text = ''
        for idx, sim_score in sorted_index_value_lst:
            turn: Turn = self.history[idx]
            # 判断一下长度
            cur = turn.user_sys_text.strip()
            use_summary = False
            if turn.content_tokens_length > 200 and shorten_history:
                use_summary = True
                cur = turn.summ.strip()

            LOCAL_CHAT_LOGGER.info(f'\n@@@@@@@@@@@@@@@@@@')
            LOCAL_CHAT_LOGGER.info(f'检索到的历史轮[使用摘要?{use_summary}]：{cur.strip()}')
            LOCAL_CHAT_LOGGER.info(f'相似度：{sim_score}')
            retrieve_history_text += f'{cur}\n\n'
        
        return retrieve_history_text
    
    def _similarity(self, v1, v2):
        vec1 = torch.FloatTensor(v1)
        vec2 = torch.FloatTensor(v2)
        cos_sim = F.cosine_similarity(vec1, vec2, dim=0)
        return cos_sim