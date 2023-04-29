# Unleashing Infinite-Length Input Capacity for Large-scale Language Models with Self-Controlled Memory System


This is the official repository for the paper ["Unleashing Infinite-Length Input Capacity for Large-scale Language Models with Self-Controlled Memory System"](https://arxiv.org/abs/2304.13343). In this paper, we introduce the Self-Controlled Memory (SCM) system to unleash infinite-length input capacity for large-scale language models.
Our SCM system is composed of three key modules: the language model agent, the memory stream, and the memory controller. 

<img src="misc/workflow.png" align="middle" width="95%">

# 🔥 Updates
- [**2023-4-26**] We released our first version [paper](https://arxiv.org/abs/2304.13343), [codes](https://github.com/wbbeyourself/SCM4LLMs). Check it out!


# 🏴󠁶󠁵󠁭󠁡󠁰󠁿 Overview

Our SCM system can be integrated with any LLMs to enable them to process ultra-long texts without any modification or fine-tuning. 

Experimental results show that our SCM system enables LLMs, which are not optimized for multi-turn dialogue, to achieve multi-turn dialogue capabilities that are comparable to ChatGPT, and to outperform ChatGPT in scenarios involving ultra-long document summarization or long-term conversations.


# ⚡️ Usage

## config

Put your openai apikey in `config/apikey.txt`, support multiple keys.

## Requirements

The key requirements are as below:

- python 3.8+
- openai 0.27.0+
- gradio 3.27.0+

Use conda to create environment.
```shell
conda create -n scm python=3.8 -y
conda activate scm
```

You can install the requirements by running:
```shell
pip install -r requirements.txt
```

## Run

Default agent model use `text-davinci-003`.

You can specify model by `--model_name`, current support model list: 
- `text-davinci-003`
- `gpt-3.5-turbo`
- other LLMs, which can understand instructions, in progress: `Alpaca`, `Vicuna`, `ChatGLM`, etc.

Functional command during dialogue, these operations will be silently done, you can see them in terminal log output.:
- `reset` or `清空`: clear dialogue history.
- `export` or `导出`: save the dialogue history to files. 
- `roll back` or `回滚`: pop previous turn dialogue.


### Ultra-long Dialogue

```bash
python dialogue-ui-demo.py
```


### Ultra-long Document Summarization


```bash
python summary-ui-demo.py
```

Additional functional command for summarization is `final summary` to get the final summary.

## TODO

- [ ] Evaluation test set
- [ ] Comparison with other models and methods



## Limitations & Risks

> A lack of appropriate datasets for evaluating the handling of extremely lengthy texts has resulted in our model being validated solely through manual verification. This method, however, is inadequate for evaluating different scenarios comprehensively and objectively. Therefore, we aim to construct a specific test set that incorporates various key indicators essential for processing long texts in diverse settings.This test set will be accompanied by a manual evaluation standard to enable a more equitable comparison with relevant methods. Moreover, we will assess the efficacy of our system on more open-source models that possess single-turn instruction comprehension capability.


> Our system has the capability to attach to any LLMs, which may be prone to factual errors, delusions, toxic language, and malicious responses. Consequently, we restrict the usage of our system to academic research purposes for now.


# 💬 Citation
If you find our work is helpful, please cite as:
```
@article{liang2023unleashing,
      title={Unleashing Infinite-Length Input Capacity for Large-scale Language Models with Self-Controlled Memory System}, 
      author={Xinnian Liang and Bing Wang and Hui Huang and Shuangzhi Wu and Peihao Wu and Lu Lu and Zejun Ma and Zhoujun Li},
      year={2023},
      eprint={2304.13343}
}
```

# 👍 Contributing

We welcome contributions and suggestions!
