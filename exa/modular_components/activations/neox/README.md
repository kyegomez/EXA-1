# Neo-X Activation Function 🚀🌌
Introducing the Neox activation function, a cutting-edge, space-themed activation function inspired by the wonders of the cosmos! Neox utilizes Fractorial Calculus to create an ultra-efficient activation function for multi-modality models. Get ready to take your neural networks to new heights and explore the vast universe of possibilities! 🌠

Features 🌟
Combines the power of Fractorial Calculus with popular base activation functions (e.g., ReLU, GELU, Swish, etc.)
Flexible and easy-to-use in your PyTorch models
Aims to improve convergence speed and generalization performance
Unlocks a new world of experimentation with fractional derivative orders
Perfect for space enthusiasts, math lovers, and AI researchers alike! 🪐
Installation 🛰️
To start using the Neox activation function, simply clone this repository:

bash
Copy code
git clone https://github.com/yourusername/neox-activation-function.git
Then, import the FractionalActivation class and other required functions into your project:

python
Copy code
from neox_activation_function import FractionalActivation, fractional_derivative
Usage 🌌
Using Neox in your neural network is as easy as instantiating the FractionalActivation class and adding it to your model architecture. Here's a quick example:

python
Copy code
import torch
import torch.nn as nn
from neox_activation_function import FractionalActivation, fractional_derivative

class MyAwesomeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, base_activation, derivative_order):
        super(MyAwesomeModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = FractionalActivation(base_activation, derivative_order)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x

# Example parameters
input_size = 784
hidden_size = 128
output_size = 10
base_activation = torch.relu
derivative_order = 0.5

# Instantiate your model with Neox
model = MyAwesomeModel(input_size, hidden_size, output_size, base_activation, derivative_order)
Contributing 🌠
We welcome all space explorers, mathematicians, and AI enthusiasts to contribute to the Neox activation function! Feel free to open issues, submit pull requests, and share your ideas on how to make Neox even more stellar! 💫

License 📜
Neox Activation Function is released under the MIT License.

Embark on this cosmic journey with us and discover the power of Neox! 🚀 Let's revolutionize activation functions together and take our neural networks to the stars! ✨