import torch
import torch.nn as nn

def fractional_derivative(x, base_activation, derivative_order, h=1e-5):
    # Apply base activation function on x
    base = base_activation(x)

    # Apply base activation function on x + h
    base_plus_h = base_activation(x + h)

    # Compute the fractional derivative using Gruenwald-Letnikov definition
    fractional_derivative = ((base_plus_h - base) / h) ** derivative_order

    return fractional_derivative


class FractionalActivation(nn.Module):
    def __init__(self, base_activation, derivative_order):
        super(FractionalActivation, self).__init__()
        self.base_activation = base_activation
        self.derivative_order = derivative_order

    def forward(self, x):
        # Compute the base activation function
        base = self.base_activation(x)

        # Compute the fractional derivative
        fractional_derivative = fractional_derivative(x, self.base_activation, self.derivative_order)

        # Combine the base activation function with its fractional derivative (e.g., addition)
        output = base + fractional_derivative

        return output



class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, base_activation, derivative_order):
        super(SimpleNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = FractionalActivation(base_activation, derivative_order)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x
    
input_size = 784  # Example for MNIST dataset
hidden_size = 128
output_size = 10
base_activation = torch.relu
derivative_order = 0.5  # Example fractional order

network = SimpleNetwork(input_size, hidden_size, output_size, base_activation, derivative_order)