import torch
import torch.nn as nn
import math
from scipy.special import gamma as scipy_gamma
import functools
import torch.jit

def gamma(n):
    return math.gamma(n)



def relu_activation(x):
    return torch.relu(x)

def factorial_tensor(tensor):
    return torch.tensor([math.factorial(int(k_i)) for k_i in tensor])

@torch.jit.script
def caputo_approximation(x, derivative_order, h, n):
    k = torch.arange(n).float().view(1, -1)
    x_expanded = x.view(-1, 1)
    h_expanded = h.view(-1, 1)
    
    # Compute the factorial of each element in k
    factorial_k = torch.empty_like(k)
    for i in range(k.shape[1]):
        factorial_k[0, i] = math.factorial(int(k[0, i]))
    
    term = ((-1)**k) * torch.exp(torch.lgamma(derivative_order + k + 1)) / (factorial_k * torch.exp(torch.lgamma(derivative_order + 1))) * (relu_activation(x_expanded - k * h_expanded) - relu_activation(x_expanded - (k + 1) * h_expanded))
    sum_terms = torch.sum(term, dim=-1)
    return sum_terms / h


class CaputoFractionalActivation(nn.Module):
    def __init__(self, base_activation, derivative_order, n=2):
        super(CaputoFractionalActivation, self).__init__()
        self.base_activation = base_activation
        self.derivative_order = torch.tensor(float(derivative_order))  # Convert derivative_order to a tensor
        self.n = torch.tensor(n)  # Convert n to a tensor


    def adaptive_step_size(self, x, min_step=1e-6, max_step=1e-3):
        x_mean = torch.mean(x)
        x_std = torch.std(x)
        step_size = min_step + (x - x_mean) / x_std * (max_step - min_step)
        return step_size

    def forward(self, x):
        h = self.adaptive_step_size(x)

        # Compute the base activation function
        base = self.base_activation(x)

        # Compute the Caputo approximate fractional derivative
        fractional_derivative = caputo_approximation(x, self.derivative_order, h.view(-1), self.n)

        # Combine the base activation function with its fractional derivative (e.g., addition)
        output = base.view_as(fractional_derivative) + fractional_derivative

        return output.view_as(x)