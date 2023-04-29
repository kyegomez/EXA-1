import torch
import torch.nn as nn
import math
from scipy.special import gamma as scipy_gamma
import functools
import torch.jit

def gamma(n):
    return math.gamma(n)

@torch.jit.script
# def caputo_approximation(x, base_activation, derivative_order, h, n):
#     k = torch.arange(n).float()
#     x_expanded = x.view(-1, 1)
#     h_expanded = h.view(-1, 1)
#     term = ((-1)**k) * scipy_gamma(derivative_order + k + 1) / (torch.tensor([math.factorial(int(k_i)) for k_i in k]) * scipy_gamma(derivative_order + 1)) * (base_activation(x_expanded - k * h_expanded) - base_activation(x_expanded - (k + 1) * h_expanded))
#     sum_terms = torch.sum(term, dim=-1)
#     return sum_terms / h

# def caputo_approximation(x, base_activation, derivative_order, h, n):
#     k = torch.arange(n).float().view(1, -1)
#     x_expanded = x.view(-1, 1)
#     h_expanded = h.view(-1, 1)
#     term = ((-1)**k) * scipy_gamma(derivative_order + k + 1) / (torch.tensor([math.factorial(int(k_i)) for k_i in k]) * scipy_gamma(derivative_order + 1)) * (base_activation(x_expanded - k * h_expanded) - base_activation(x_expanded - (k + 1) * h_expanded))
#     sum_terms = torch.sum(term, dim=-1)
#     return sum_terms / h

def caputo_approximation(x, base_activation, derivative_order, h, n):
    k = torch.arange(n).float().view(1, -1)
    x_expanded = x.view(-1, 1)
    h_expanded = h.view(-1, 1)
    term = ((-1)**k) * torch.exp(torch.lgamma(derivative_order + k + 1)) / (torch.tensor([math.factorial(int(k_i)) for k_i in k]) * torch.exp(torch.lgamma(derivative_order + 1))) * (base_activation(x_expanded - k * h_expanded) - base_activation(x_expanded - (k + 1) * h_expanded))
    sum_terms = torch.sum(term, dim=-1)
    return sum_terms / h

class CaputoFractionalActivation(nn.Module):
    def __init__(self, base_activation, derivative_order, n=5):
        super(CaputoFractionalActivation, self).__init__()
        self.base_activation = base_activation
        self.derivative_order = derivative_order
        self.n = n

    @functools.lru_cache(maxsize=None)
    def memoized_base_activation(self, x):
        return self.base_activation(x)

    def adaptive_step_size(self, x, min_step=1e-6, max_step=1e-3):
        x_mean = torch.mean(x)
        x_std = torch.std(x)
        step_size = min_step + (x - x_mean) / x_std * (max_step - min_step)
        return step_size

    # def forward(self, x):
    #     h = self.adaptive_step_size(x)

    #     # Compute the base activation function
    #     base = self.memoized_base_activation(x)

    #     # Compute the Caputo approximate fractional derivative
    #     fractional_derivative = caputo_approximation(x, self.memoized_base_activation, self.derivative_order, h.view(-1), self.n)

    #     # Combine the base activation function with its fractional derivative (e.g., addition)
    #     output = base + fractional_derivative

    #     return output.view_as(x)
    def forward(self, x):
        h = self.adaptive_step_size(x)

        # Compute the base activation function
        base = self.memoized_base_activation(x)

        # Compute the Caputo approximate fractional derivative
        fractional_derivative = caputo_approximation(x, self.memoized_base_activation, self.derivative_order, h.view(-1), self.n)

        # Combine the base activation function with its fractional derivative (e.g., addition)
        output = base.view_as(fractional_derivative) + fractional_derivative

        return output.view_as(x)

