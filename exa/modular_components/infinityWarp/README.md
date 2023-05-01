# Infinity Warp

![InfinityWarp Conduct Inference While Training](/InfinityWarp.png)


Infinity Warp is a flexible and extensible framework that enables seamless integration of custom training and inference functions for deep learning models. It leverages the power of `torch.multiprocessing` to perform concurrent training and inference, while ensuring minimal interference between the two processes.

## Features
- Easy integration with your custom training and inference functions
- Concurrent execution of training and inference operations
- Minimal interference between training and inference processes
- Supports various deep learning models and tasks

## Installation

To use Infinity Warp, simply clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/infinity-warp.git
```

## Usage

Here's an example of how to use Infinity Warp with your custom training and inference functions:

```python
from infinity_warp import InfinityWarp

# Define your custom training and inference functions
def my_train_fn(model, train_data, train_labels):
    # Your custom training function logic here

def my_infer_fn(model, infer_data):
    # Your custom inference function logic here
    return predictions

# Instantiate the InfinityWarp class with your model, data, and functions
iw = InfinityWarp(
    model=my_model,
    train_data=my_train_data,
    train_labels=my_train_labels,
    infer_data=my_infer_data,
    train_fn=my_train_fn,
    infer_fn=my_infer_fn
)

# Start the concurrent training and inference processes
iw.start()
```

## Contributing

We welcome contributions to improve Infinity Warp! If you'd like to contribute, please follow these steps:

1. Fork the repository on GitHub
2. Clone your forked repository to your local machine
3. Make your changes and commit them to your fork
4. Create a pull request to the original repository

We appreciate your help in making Infinity Warp better!

## License

Infinity Warp is released under the [GNU GENERAL PUBLIC LICENSE](LICENSE).