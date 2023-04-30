# An Awesome List for AI-Training!

# Distributed:

## Deepspeed

https://huggingface.co/docs/accelerate/usage_guides/deepspeed

ZeRO-Offload: Democratizing Billion-Scale Model Training: https://arxiv.org/abs/2101.06840

ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning: https://arxiv.org/abs/2104.07857

ZeRO: Memory Optimizations Toward Training Trillion Parameter Models: https://arxiv.org/abs/1910.02054

code: https://github.com/microsoft/DeepSpeed

## Eleuter/DeeperSpeed:

repo: https://github.com/EleutherAI/DeeperSpeed

## HuggingFace Accelerate:

🤗 Accelerate is a library that enables the same PyTorch code to be run across any distributed configuration by adding just four lines of code! In short, training and inference at scale made simple, efficient and adaptable.

https://huggingface.co/docs/accelerate/index

Fully Sharded Data Parallel

To accelerate training huge models on larger batch sizes, we can use a fully sharded data parallel model. This type of data parallel paradigm enables fitting more data and larger models by sharding the optimizer states, gradients and parameters. To read more about it and the benefits, check out the Fully Sharded Data Parallel blog. We have integrated the latest PyTorch’s Fully Sharded Data Parallel (FSDP) training feature. All you need to do is enable it through the config.

https://huggingface.co/docs/accelerate/usage_guides/fsdp

## AutoTrain:

🤗 AutoTrain is a no-code tool for training state-of-the-art models for Natural Language Processing (NLP) tasks, for Computer Vision (CV) tasks, and for Speech tasks and even for Tabular tasks. It is built on top of the awesome tools developed by the Hugging Face team, and it is designed to be easy to use.

https://huggingface.co/docs/autotrain/index

## Onnx Runtime:

ONNX Runtime inference can enable faster customer experiences and lower costs, supporting models from deep learning frameworks such as PyTorch and TensorFlow/Keras as well as classical machine learning libraries such as scikit-learn, LightGBM, XGBoost, etc. ONNX Runtime is compatible with different hardware, drivers, and operating systems, and provides optimal performance by leveraging hardware accelerators where applicable alongside graph optimizations and transforms. Learn more →

code: https://github.com/microsoft/onnxruntime

docs: https://onnxruntime.ai/docs/

## NVIDIA APEX:

A PyTorch Extension: Tools for easy mixed precision and distributed training in Pytorch

github: https://github.com/NVIDIA/apex

docs: https://nvidia.github.io/apex/

https://github.com/microsoft/DeepSpeed

## Nvidia DALI:

A GPU-accelerated library containing highly optimized building blocks and an execution engine for data processing to accelerate deep learning training and inference applications.

REPO: https://github.com/NVIDIA/DALI

## ColossalAI:

repo: https://github.com/hpcaitech/ColossalAI

https://colossalai.org/

# Reinforcement:

carperai/trlx:

trlX is a distributed training framework designed from the ground up to focus on fine-tuning large language models with reinforcement learning using either a provided reward function or a reward-labeled dataset.

https://github.com/CarperAI/trlx

### TRL - Transformer Reinforcement Learning

repo: https://github.com/lvwerra/trl/

# Efficiency:

LoRA: Low-Rank Adaptation of Large Language Models

paper: https://arxiv.org/abs/2106.09685

code: https://github.com/microsoft/LoRA


# Language:

## Triton by openai:

An Intermediate Language and Compiler for Tiled Neural Network Computations

repo: https://github.com/openai/triton

paper: http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf

## Jax:

Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more

code: https://github.com/google/jax

docs: https://jax.readthedocs.io/en/latest/

# Compilers:

## Hidet

Hidet is an open-source deep learning compiler, written in Python. It supports end-to-end compilation of DNN models from PyTorch and ONNX to efficient cuda kernels. A series of graph-level and operator-level optimizations are applied to optimize the performance.

code: https://github.com/hidet-org/hidet

docs: https://docs.hidet.org/stable/index.html

# Quantization:

LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale: https://arxiv.org/abs/2208.07339

## GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers: https://arxiv.org/abs/2210.17323

code: https://github.com/IST-DASLab/gptq

## GPTQ-for-LLaMa:

GPTQ is SOTA one-shot weight quantization method

CODE: https://github.com/qwopqwop200/GPTQ-for-LLaMa

## bitsandbytes:

The bitsandbytes is a lightweight wrapper around CUDA custom functions, in particular 8-bit optimizers, matrix multiplication (LLM.int8()), and quantization functions.

repo: https://github.com/TimDettmers/bitsandbytes

## AutoGPTQ

An easy-to-use model quantization package with user-friendly apis, based on GPTQ algorithm.

https://github.com/PanQiWei/AutoGPTQ

# Frameworks:

Ray: 

https://github.com/ray-project/ray

lightning: 

Deep learning framework to train, deploy, and ship AI products Lightning fast.

https://github.com/Lightning-AI/lightning




# Model Tricks

## U-ViT
💡This codebase supports useful techniques for efficient training and sampling of diffusion models:

Mixed precision training with the huggingface accelerate library (🥰automatically turned on)
Efficient attention computation with the facebook xformers library (needs additional installation)
Gradient checkpointing trick, which reduces ~65% memory (🥰automatically turned on)
With these techniques, we are able to train our largest U-ViT-H on ImageNet at high resolutions such as 256x256 and 512x512 using a large batch size of 1024 with only 2 A100❗

We highly suggest install xformers, which would greatly speed up the attention computation for both training and inference.