import torch
import torch.nn.functional as F
import numpy as np
from api import *
import tiktoken
from tools import *
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


class SummaryTurn(object):

    def __init__(self, paragraph, summary, embedding):
        self.paragraph = paragraph
        self.summary = summary
        self.embedding = embedding
        self.content_tokens_length = 0
        self.summary_tokens_length = 0
    
    def to_json(self):
        js = {
            'paragraph': self.paragraph,
            'summary': self.summary
        }
        return js
    
    def to_json_str(self):
        js = self.to_json()
        js_str = json.dumps(js, ensure_ascii=False) + '\n'
        return js_str
    
    def to_plain_text(self):
        text = f'[paragraph]:\n{self.paragraph}\n\n'
        text += f'[summary]:\n{self.summary}\n'
        text += ('-' * 30 + '\n\n')
        return text


class SummaryBot(object):

    def __init__(self, model_name) -> None:
        assert model_name in MODEL_LIST, f'model name "{model_name}" is not in {MODEL_LIST}'
        self.model_name = model_name
        self.api_func = MODEL_MAP[model_name]
        self.turbo_func = MODEL_MAP['gpt-3.5-turbo']
        self.embedding_func = MODEL_EMBEDDING_MAP[model_name]
        self.tokenize_func = get_tokenizer_func(self.model_name)
        self.history: list[SummaryTurn] = []
        self.final_summary = ''

    def clear_history(self):
        self.history = []
    
    def roll_back(self):
        self.history.pop()
    
    def export_history(self):
        hist_lst = [one.to_json() for one in self.history]
        hist_lst.append({'final summary': self.final_summary})

        hist_txt_lst = [one.to_plain_text() for one in self.history]
        hist_txt_lst.append(f"final summary: \n\n{self.final_summary}\n\n")
        stamp = datetime2str()
        json_filename = f'history/summary-{hist_lst[0]["summary"][:10]}-{self.model_name}-{stamp}.json'
        txt_filename = f'history/summary-{hist_lst[0]["summary"][:10]}-{self.model_name}-{stamp}.txt'
        save_json_file(json_filename, hist_lst)
        save_file(txt_filename, hist_txt_lst)
    

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

    def _summarize_paragraphs(self, paragraphs):
        output = ''
        templates_zh = '给定一个文档的各个段落摘要，请写出该文档的最终摘要，要求：(1) 用一段文字提取各段摘要里的关键信息，去除重复信息，组织成逻辑通顺的文本; (2) 字数不超过1500字; (3) 摘要内容使用中文。\n\n各段落摘要：\n\n{}\n\n文档摘要：'
        templates_en = 'Given the summaries of each paragraph of a document, please write the final summary of the document, with the following requirements: (1) extract the key information from each paragraph summary into a single paragraph, removing duplicate information and organizing it into a logically coherent text; (2) the word count does not exceed 1500 words; (3) the summary content is in English.\n\nSummaries of each paragraph:\n\n{}\n\nDocument Summarization:'

        paragraphs_text = '\n\n'.join(paragraphs).strip()

        lang2template = {
            LANG_EN: templates_en,
            LANG_ZH: templates_zh
        }

        tmp = choose_language_template(lang2template, paragraphs[0])
        input_text = tmp.format(paragraphs_text)
        LOCAL_CHAT_LOGGER.info(f"input_text:\n\n{input_text}")
        output = self.ask(input_text)
        LOCAL_CHAT_LOGGER.info(f"output:\n\n{output}")
        return output

    
    def _divide_conquer_summary(self, content_lst):
        tgt_summary = ''
        summary_token_length_lst = [len(self.tokenize_func(txt)) for txt in content_lst]
        LOCAL_CHAT_LOGGER.info(f"summary_token_length_lst:\n\n{summary_token_length_lst}")
        total_tokens = sum(summary_token_length_lst)

        def split_array(arr):
            mid = len(arr) // 2
            return arr[:mid], arr[mid:]

        if total_tokens < 2500:
            tgt_summary = self._summarize_paragraphs(content_lst)
        else:
            left, right = split_array(content_lst)
            left_summary = self._divide_conquer_summary(left)
            right_summary = self._divide_conquer_summary(right)
            tgt_summary = self._divide_conquer_summary([left_summary, right_summary])
        LOCAL_CHAT_LOGGER.info(f"tgt_summary:\n\n{tgt_summary}")
        return tgt_summary

    def get_final_summary(self):
        sub_summary_lst = [item.summary for item in self.history]
        final_summary = self._divide_conquer_summary(sub_summary_lst)
        self.final_summary = final_summary
        return final_summary
    
    def add_turn_history(self, turn: SummaryTurn):
        turn.content_tokens_length = len(self.tokenize_func(turn.paragraph))
        turn.summary_tokens_length = len(self.tokenize_func(turn.summary))
        self.history.append(turn)
    
    def get_turn_for_previous(self):
        turn = self.history[-1]
        if turn.content_tokens_length < 500:
            return turn.paragraph
        else:
            return turn.summary
    

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

        index_value_lst = [(idx, v) for idx, v in zip(topk_indices, topk_values)]
        # print(index_value_lst)
        sorted_index_value_lst = sorted(index_value_lst, key=lambda x: x[0])
        LOCAL_CHAT_LOGGER.info(f'\n--------------\n')
        LOCAL_CHAT_LOGGER.info(f"\nTop{k}相似历史索引及其相似度: \n\n{sorted_index_value_lst}\n")
        LOCAL_CHAT_LOGGER.info(f'\n--------------\n')


        retrieve_history_text = ''
        for idx, sim_score in sorted_index_value_lst:
            turn: SummaryTurn = self.history[idx]
            # 判断一下长度
            cur = turn.paragraph.strip()
            use_summary = False
            if turn.content_tokens_length > 300:
                use_summary = True
                cur = turn.summary.strip()

            LOCAL_CHAT_LOGGER.info(f'\n@@@@@@@@@@@@@@@@@@')
            LOCAL_CHAT_LOGGER.info(f'检索到的历史轮[使用摘要?{use_summary}]：{cur.strip()}')
            LOCAL_CHAT_LOGGER.info(f'相似度：{sim_score}')
            retrieve_history_text += f'{cur}\n\n'
        
        return retrieve_history_text.strip()
    
    def _similarity(self, v1, v2):
        vec1 = torch.FloatTensor(v1)
        vec2 = torch.FloatTensor(v2)
        cos_sim = F.cosine_similarity(vec1, vec2, dim=0)
        return cos_sim