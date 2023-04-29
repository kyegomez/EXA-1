import torch
import torch.nn as nn
import math

def gamma(n):
    return math.gamma(n)

def caputo_approximation(x, base_activation, derivative_order, h, n):
    sum_terms = 0.0
    for k in range(n):
        term = ((-1)**k) * gamma(derivative_order + k + 1) / (math.factorial(k) * gamma(derivative_order + 1)) * (base_activation(x - k * h) - base_activation(x - (k + 1) * h))
        sum_terms += term
    return sum_terms / h

class CaputoFractionalActivation(nn.Module):
    def __init__(self, base_activation, derivative_order, n=10):
        super(CaputoFractionalActivation, self).__init__()
        self.base_activation = base_activation
        self.derivative_order = derivative_order
        self.n = n

    def forward(self, x):
        h = 1e-5  # You can use the adaptive step size function from the previous examples to get h

        # Compute the base activation function
        base = self.base_activation(x)

        # Compute the Caputo approximate fractional derivative
        fractional_derivative = caputo_approximation(x, self.base_activation, self.derivative_order, h, self.n)

        # Combine the base activation function with its fractional derivative (e.g., addition)
        output = base + fractional_derivative

        return output


        
        