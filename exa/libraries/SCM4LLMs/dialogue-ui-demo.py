import os
import sys
import argparse
from os.path import join
from tools import *
import logging
from api import set_api_logger
from chat import ChatBot, Turn, set_chat_logger
import gradio as gr

args: argparse.Namespace = None
bot: ChatBot = None


def summarize_embed_one_turn(bot: ChatBot, dialogue_text, dialogue_text_with_index):
    lang2template = {
        LANG_EN: 'Below is a conversation between a user and an AI assistant. Please write a summary for each of them in one sentence and list them in separate paragraphs, while trying to preserve the key information of the user’s question and the assistant’s answer as much as possible. \n\nconversation content: \n\n{}\n\nSummary:',
        LANG_ZH: '以下是用户和人工智能助手的一段对话，请分别用一句话写出用户摘要、助手摘要，分段列出，要求尽可能保留用户问题和助手回答的关键信息。\n\n对话内容：\n\n{}\n\n摘要：'
    }

    tmp = choose_language_template(lang2template, dialogue_text)
    input_text = tmp.format(dialogue_text)
    logger.info(f'turn summarization input_text: \n\n{input_text}')
    # 如果原文很短，保留原文即可
    summarization = input_text
    if get_token_count_davinci(input_text) > 300:
        logger.info(f'current turn text token count > 300, summarize !\n\n')
        summarization = bot.ask(input_text)
        logger.info(f'Summarization is:\n\n{summarization}\n\n')
    embedding = bot.vectorize(dialogue_text_with_index)
    return summarization, embedding

# todo: 这部分长度可能会超长，需要动态设置一下。
def get_concat_input(user_str, pre_sre, hist_str=None):
    templates_no_hist_zh = '以下是用户和人工智能助手的对话，请根据历史对话内容，回答用户当前问题：\n\n上一轮对话：\n\n{}\n\n###\n\n用户：{}\n\n助手：'
    templates_no_hist_en = 'The following is a conversation between a user and an AI assistant. Please answer the current question based on the history of the conversation:\n\nPrevious conversation:\n\n{}\n\n###\n\nUser: {}\n\nAssistant:'

    lang2template = {
        LANG_EN: templates_no_hist_en,
        LANG_ZH: templates_no_hist_zh
    }

    templates_no_hist = choose_language_template(lang2template, user_str)

    templates_hist_zh = '以下是用户和人工智能助手的对话，请根据历史对话内容，回答用户当前问题：\n\n相关历史对话：\n\n{}\n\n上一轮对话：\n\n{}\n\n###\n\n用户：{}\n\n助手：'
    templates_hist_en = 'The following is a conversation between a user and an AI assistant. Please answer the current question based on the history of the conversation:\n\nRelated conversation history:\n\n{}\n\nPrevious conversation:\n\n{}\n\n###\n\nUser: {}\n\nAssistant:'

    lang2template = {
        LANG_EN: templates_hist_en,
        LANG_ZH: templates_hist_zh
    }

    templates_hist = choose_language_template(lang2template, user_str)

    if hist_str:
        input_text = templates_hist.format(hist_str, pre_sre, user_str)
    else:
        input_text = templates_no_hist.format(pre_sre, user_str)
    return input_text


def check_key_file(key_file):
    if not os.path.exists(key_file):
        print(f'[{key_file}] not found! Please put your apikey in the txt file.')
        sys.exit(-1)


def judge_need_history(user_instruction):
    templates_zh = '给定一个用户指令，判断执行该指令是否需要历史信息或者上文的信息，或者需要回忆对话内容，只需要回答是(A)或者否(B)，不需要解释信息：\n\n指令：{}'
    templates_en = 'Given a user command, determine whether executing the command requires historical or previous information, or whether it requires recalling the conversation content. Simply answer yes (A) or no (B) without explaining the information:\n\nCommand:{}'

    lang2template = {
        LANG_EN: templates_en,
        LANG_ZH: templates_zh
    }

    tmp = choose_language_template(lang2template, user_instruction)
    input_text = tmp.format(user_instruction)
    is_need = bot.is_history_need(input_text)
    logger.info(f'\n--------------\nis_need: {is_need}\n--------------\n')
    return is_need


def get_first_prompt(user_text, model_name):
    if model_name in ['gpt-3.5-turbo']:
        return user_text
    else:
        templates_zh = '假设你是人工智能助手， 请回答用户的问题和请求：\n\n用户：{}\n\n助手：'
        templates_en = 'Assuming you are an AI assistant, please answer the user\'s questions and requests:\n\nUser: {}\n\nAssistant:'

        lang2template = {
            LANG_EN: templates_en,
            LANG_ZH: templates_zh
        }

        tmp = choose_language_template(lang2template, user_text)
        concat_input = tmp.format(user_text)
        return concat_input


def my_chatbot(user_input, history):
    history = history or []

    user_input = user_input.strip()

    my_history = list(sum(history, ()))

    COMMAND_RETURN = '命令已成功执行！'

    if user_input in ['清空', 'reset']:
        # history.append((user_input, COMMAND_RETURN))
        history = []
        bot.clear_history()
        logger.info(f'[User Command]: {user_input} {COMMAND_RETURN}')
        return history, history
    elif user_input in ['导出', 'export']:
        # history.append((user_input, COMMAND_RETURN))
        bot.export_history()
        logger.info(f'[User Command]: {user_input} {COMMAND_RETURN}')
        return history, history
    elif user_input in ['回退', '回滚', 'roll back']:
        history.pop()
        bot.roll_back()
        logger.info(f'[User Command]: {user_input} {COMMAND_RETURN}')
        return history, history

    # 历史: my_history
    # 当前输入: user_input


    len_hist = len(bot.history)
    cur_turn_index = len_hist + 1
    if len_hist == 0:
        concat_input = get_first_prompt(user_input, args.model_name)
    else:
        retrieve = None
        is_need = judge_need_history(user_input)
        # 并且 需要历史信息才给
        if cur_turn_index > 2 and is_need:
            retrieve = bot.get_related_turn(user_input, args.similar_top_k)
        
        concat_input = get_concat_input(user_input, bot.get_turn_for_previous(), hist_str=retrieve)
    
    logger.info(f'\n--------------\n[第{cur_turn_index}轮] concat_input:\n\n{concat_input}\n--------------\n')

    try:
        rsp: str = bot.ask(concat_input)
    except Exception as e:
        logger.error(f'ERROR: \n\n{e}')
        rsp = '喵呜，您的请求好像掉进了喵喵的世界里了~'
        history.append((user_input, rsp))
        return history, history

    system_text = rsp.strip()

    cur_text_without_index = '用户：{}\n\n助手：{}'.format(user_input, system_text)
    cur_text_with_index = '[第{}轮]\n\n用户：{}\n\n助手：{}'.format(cur_turn_index, user_input, system_text)

    if detect_language(user_input) == LANG_EN:
        cur_text_without_index = 'User: {}\n\nAssistant: {}'.format(user_input, system_text)
        cur_text_with_index = '[Turn {}]\n\nUser: {}\n\nAssistant: {}'.format(cur_turn_index, user_input, system_text)


    try:
        summ, embedding = summarize_embed_one_turn(bot, cur_text_without_index, cur_text_with_index)
    except Exception as e:
        logger.error(f'summarize_embed_one_turn ERROR: \n\n{e}')
        rsp = '摘要出错，喵呜，您的请求好像掉进了喵喵的世界里了~'
        history.append((user_input, rsp))
        return history, history

    cur_turn = Turn(user_input=user_input, system_response=system_text, user_sys_text=cur_text_with_index, summ=summ, embedding=embedding)
    bot.add_turn_history(cur_turn)
    
    my_history.append(user_input)
    output = system_text
    history.append((user_input, output))
    return history, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_choices = ['text-davinci-003', 'gpt-3.5-turbo', 'bloom', 'alpaca', 'llama']
    parser.add_argument("--apikey_file", type=str, default="./config/apikey.txt")
    parser.add_argument("--model_name", type=str, default="text-davinci-003", choices=model_choices)
    parser.add_argument("--target_file", type=str)
    parser.add_argument("--logfile", type=str, default="./logs/log.txt")
    parser.add_argument("--history_file", type=str)
    parser.add_argument("--similar_top_k", type=int, default=4)
    args = parser.parse_args()

    check_key_file(args.apikey_file)

    log_path = args.logfile
    makedirs(log_path)
    # 配置日志记录

    logger = logging.getLogger('dialogue_logger')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('【%(asctime)s - %(levelname)s】 - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    set_chat_logger(logger)
    set_api_logger(logger)

    logger.info('\n\n\n')
    logger.info('#################################')
    logger.info('#################################')
    logger.info('#################################')
    logger.info('\n\n\n')
    logger.info(f"args: \n\n{args}\n")

    stamp = datetime2str()
    # print(stamp)

    if args.target_file:
        history_file = f'{args.target_file}'
    else:
        history_file = f'./history/{stamp}.json'
    embedding_file = history_file + '.emb.json'
    
    bot = ChatBot(model_name=args.model_name)
    if args.history_file:
        history_file = args.history_file
        embedding_file = history_file + '.emb.json'
        bot.load_history(args.history_file)
    
    makedirs(history_file)
    makedirs(embedding_file)

    # if args.target_file:
    #     with open(history_file, 'w') as file: pass
    #     with open(embedding_file, 'w') as file: pass

    with gr.Blocks() as demo:
        gr.Markdown(f"<h1><center>Long Dialogue Chatbot ({args.model_name})</center></h1>")
        chatbot = gr.Chatbot()
        state = gr.State()
        txt = gr.Textbox(show_label=False, placeholder="Ask me a question and press enter.").style(container=False)
        txt.submit(my_chatbot, inputs=[txt, state], outputs=[chatbot, state])
        
    demo.launch(share = True)