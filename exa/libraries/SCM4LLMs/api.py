import os
import openai
import torch
import time
from tools import get_lines, time_cost, append_file
from transformers import AutoTokenizer, AutoModelForCausalLM

BLOOM_MODEL = None
BLOOM_TOKENIZER = None

LOCAL_API_LOGGER = None
def set_api_logger(one):
    global LOCAL_API_LOGGER
    LOCAL_API_LOGGER = one

class KeyManager(object):

    index_save_file = '.key.index'

    def __init__(self, filename) -> None:
        self.keys = get_lines(filename)
        self.key_index = 0
        if os.path.exists(self.index_save_file):
            index = int(get_lines(self.index_save_file)[-1])
            index += 1
            index %= len(self.keys)
            self.key_index = index

    def get_api_key(self):
        self.key_index += 1
        self.key_index %= len(self.keys)
        append_file(self.index_save_file, [str(self.key_index)+'\n'])
        cur_key = self.keys[self.key_index]
        print(f'\n-----------------\nkey: {cur_key}\nindex:{self.key_index}\n-----------------\n')
        return cur_key

def get_initialized_hf_model(path):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


KEY_MANAGER = KeyManager('config/apikey.txt')


def call_embedding_openai(text):
    openai.api_key = KEY_MANAGER.get_api_key()
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    embedding = response['data'][0]['embedding']
    return embedding


def call_embedding_bloom(text):
    global BLOOM_MODEL
    global BLOOM_TOKENIZER
    checkpoint_path = '/your/checkpoint/path'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not BLOOM_MODEL:
        t0 = time.time()
        LOCAL_API_LOGGER.info('Loading bloom model ... Please wait a few minutes ...')
        BLOOM_MODEL, BLOOM_TOKENIZER = get_initialized_hf_model(checkpoint_path)
        BLOOM_MODEL.to(device)
        LOCAL_API_LOGGER.info('Model Loaded Success !!!')
        time_cost(t0)
    
    model, tokenizer = BLOOM_MODEL, BLOOM_TOKENIZER
    input_ids = tokenizer.encode(text, return_tensors='pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)

    with torch.no_grad():
        model_output = model(input_ids, output_hidden_states=True, return_dict=True)

    # 获取嵌入向量
    last_hidden_state = model_output.hidden_states
    LOCAL_API_LOGGER.info(f'len last_hidden_state: {len(last_hidden_state)}')
    # 获取最后一个 token 的嵌入
    last_indx = input_ids.size()[1] - 1
    if last_indx == 0:
        last_token_embedding = last_hidden_state[-1].squeeze()
    else:
        last_token_embedding = last_hidden_state[-1].squeeze()[last_indx].squeeze()
    LOCAL_API_LOGGER.info(f'last_token_embedding len: {len(last_token_embedding)}')
    # print(f'last_token_embedding[:4] : {last_token_embedding[:3]}')
    last_token_embedding = last_token_embedding.tolist()
    return last_token_embedding


def call_text_davinci_003(prompt):
    api_model_index = 'text-davinci-003'
    openai.api_key = KEY_MANAGER.get_api_key()
    response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=1024,
            stop=["\n\n\n", "###"],
        )
    LOCAL_API_LOGGER.info(f"[{api_model_index} request cost token]: {response['usage']['total_tokens']}")
    LOCAL_API_LOGGER.info(f"[{api_model_index} available tokens]: {4000 - response['usage']['total_tokens']}")
    text = response['choices'][0]['text'].strip()
    return text


def call_gpt3_5_turbo(prompt):
    api_model_index = 'gpt-3.5-turbo'
    openai.api_key = KEY_MANAGER.get_api_key()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        stop=["###"]
    )
    LOCAL_API_LOGGER.info(f"[{api_model_index} request cost token]: {response['usage']['total_tokens']}")
    LOCAL_API_LOGGER.info(f"[{api_model_index} available tokens]: {4000 - response['usage']['total_tokens']}")
    text = response['choices'][0]['message']['content'].strip()
    return text


def call_bloom(prompt):
    print(f'call_bloom : \n\nprompt \n\n{prompt}')
    global BLOOM_MODEL
    global BLOOM_TOKENIZER
    checkpoint_path = '/mnt/bn/slp-llm/sft_lxn/bloom-alpaca/bloomz-alpaca-chat+data0407-allin-bz1k_epoch2_lr3e-6_global_step11364_hf'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not BLOOM_MODEL:
        t0 = time.time()
        LOCAL_API_LOGGER.info('Loading bloom model ... Please wait a few minutes ...')
        BLOOM_MODEL, BLOOM_TOKENIZER = get_initialized_hf_model(checkpoint_path)
        BLOOM_MODEL.to(device)
        LOCAL_API_LOGGER.info('Model Loaded Success !!!')
        time_cost(t0)
    
    model, tokenizer = BLOOM_MODEL, BLOOM_TOKENIZER
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)

    LOCAL_API_LOGGER.info('generating ...')
    max_new_tokens = min(512, 2000 - len(input_ids))
    LOCAL_API_LOGGER.info(f'len input_ids = {len(input_ids[0])}')
    LOCAL_API_LOGGER.info(f'max_new_tokens: {max_new_tokens}')
    outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample = True, top_k = 30, top_p = 0.85, temperature = 0.5, repetition_penalty=1., eos_token_id=2, bos_token_id=1, pad_token_id=0)
    rets = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    LOCAL_API_LOGGER.info('generating done!')
    
    text = rets[0].strip().replace(prompt, "")
    return text


def call_alpaca():
    pass


def call_llama():
    pass


MODEL_MAP = {
    'text-davinci-003': call_text_davinci_003,
    'gpt-3.5-turbo': call_gpt3_5_turbo,
    'bloom': call_bloom,
    'alpaca': call_alpaca,
    'llama': call_llama
}

MODEL_EMBEDDING_MAP = {
    'text-embedding-ada-002': call_embedding_openai,
    'text-davinci-003': call_embedding_openai,
    'gpt-3.5-turbo': call_embedding_openai,
    'bloom': call_embedding_bloom,
    'alpaca': call_alpaca,
    'llama': call_llama
}

MODEL_LIST = [k for k in MODEL_MAP.keys()]

