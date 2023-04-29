import os
import sys
import argparse
from os.path import join
from tools import *
import logging
from api import set_api_logger
from summary import SummaryBot, SummaryTurn, set_chat_logger
import gradio as gr

args: argparse.Namespace = None
bot: SummaryBot = None


# todo: 这部分长度可能会超长，需要动态设置一下。
def get_concat_input(user_str, pre_sre, hist_str=None):
    templates_no_hist_zh = '给定当前文本和上文内容，请写出当前文本的摘要，要求：1）将上文内容作为当前文本的背景信息; 2）对当前文本进行压缩; 3) 输出内容使用中文：\n\n上文内容：{}\n\n当前文本：{}\n\n摘要：'
    templates_no_hist_en = 'Given the current text and the previous text, please provide a summary of the current text. The requirements are: 1) use the previous text as background information for the current text; 2) compress the current text; 3) output the summary in English.\n\nPrevious text: {}\n\nCurrent text: {}\n\nSummary:'

    lang2template = {
        LANG_EN: templates_no_hist_en,
        LANG_ZH: templates_no_hist_zh
    }

    templates_no_hist = choose_language_template(lang2template, user_str)

    templates_hist_zh = '给定当前文本和上文内容，请写出当前文本的摘要，要求：1）将上文内容作为当前文本的背景信息; 2）对当前文本进行压缩; 3) 输出内容使用中文：\n\n上文内容：{}\n\n{}\n\n当前文本：{}\n\n摘要：'
    templates_hist_en = 'Given the current text and the previous text, please provide a summary of the current text. The requirements are: 1) use the previous text as background information for the current text; 2) compress the current text; 3) output the summary in English.\n\nPrevious text: {}\n\n{}\n\nCurrent text: {}\n\nSummary:'

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
    templates_zh = '给定一段文本内容，判断对该文本进行摘要是否需要历史信息或者上文的信息，要求：(1) 回答是(A)或者否(B)，(2) 如果回答是(A)，请说明需要补充哪些信息：\n\n文本内容：{}\n\n答案：'
    templates_en = 'Given a piece of text, determine whether historical or previous information is needed for summarization. Requirements: (1) Answer with Yes(A) or No(B), (2) If the answer is Yes(A), please explain what information needs to be supplemented:\n\nText Content: {}\n\nAnswer:'

    lang2template = {
        LANG_EN: templates_en,
        LANG_ZH: templates_zh
    }

    tmp = choose_language_template(lang2template, user_instruction)
    input_text = tmp.format(user_instruction)
    is_need = bot.is_history_need(input_text)
    logger.info(f"\n--------------\n[is_need]: {'需要历史' if is_need else '不需要'}\n--------------\n")
    return is_need


def get_first_prompt(user_text, model_name):
    # todo: model specific prompt design, use [model_name]
    templates_zh = '以下文本内容是长文档的一部分，请写出文本摘要：\n\n文本内容：{}\n\n摘要：'
    templates_en = 'This is a part of a lengthy document, please write a summary:\n\nDocument content: {}\n\nSummary:'

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
    elif user_input in ['final summary', '最终摘要']:
        final_summary = bot.get_final_summary()
        history.append((user_input, final_summary))
        return history, history


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

    summary = rsp.strip()

    try:
        embedding = bot.vectorize(summary)
    except Exception as e:
        logger.error(f'bot.vectorize ERROR: \n\n{e}')
        rsp = '摘要出错，喵呜，您的请求好像掉进了喵喵的世界里了~'
        history.append((user_input, rsp))
        return history, history

    cur_turn = SummaryTurn(paragraph=user_input, summary=summary, embedding=embedding)
    bot.add_turn_history(cur_turn)
    
    history.append((user_input, f"[summary]: {summary}"))
    return history, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_choices = ['text-davinci-003', 'gpt-3.5-turbo', 'bloom', 'alpaca', 'llama']
    parser.add_argument("--apikey_file", type=str, default="./config/apikey.txt")
    parser.add_argument("--model_name", type=str, default="text-davinci-003", choices=model_choices)
    parser.add_argument("--target_file", type=str)
    parser.add_argument("--logfile", type=str, default="./logs/summary.log.txt")
    parser.add_argument("--history_file", type=str)
    parser.add_argument("--similar_top_k", type=int, default=4)
    args = parser.parse_args()

    check_key_file(args.apikey_file)

    log_path = args.logfile
    makedirs(log_path)
    # 配置日志记录

    logger = logging.getLogger('summary_logger')
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
    
    bot = SummaryBot(model_name=args.model_name)
    # if args.history_file:
    #     history_file = args.history_file
    #     embedding_file = history_file + '.emb.json'
    #     bot.load_history(args.history_file)
    
    # makedirs(history_file)
    # makedirs(embedding_file)

    # if args.target_file:
    #     with open(history_file, 'w') as file: pass
    #     with open(embedding_file, 'w') as file: pass

    with gr.Blocks() as demo:
        gr.Markdown(f"<h1><center>Long Summary Chatbot ({args.model_name})</center></h1>")
        chatbot = gr.Chatbot()
        state = gr.State()
        txt = gr.Textbox(show_label=False, placeholder="Paste me with a paragraph and press enter.").style(container=False)
        txt.submit(my_chatbot, inputs=[txt, state], outputs=[chatbot, state])
        
    demo.launch(share = True)