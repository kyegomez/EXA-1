import torch
import torch.nn as nn
import math
from scipy.special import gamma as scipy_gamma
import math
import functools

# @functools.lru_cache(maxsize=None)
# def memoized_base_activation(x):
#     return base_activation(x)



def gamma(n):
    return math.gamma(n)

# def caputo_approximation(x, base_activation, derivative_order, h, n):
#     k = torch.arange(n).float()
#     term = ((-1)**k) * scipy_gamma(derivative_order + k + 1) / (torch.tensor([math.factorial(int(k_i)) for k_i in k]) * scipy_gamma(derivative_order + 1)) * (base_activation(x - k * h) - base_activation(x - (k + 1) * h))
#     sum_terms = torch.sum(term, dim=-1)
#     return sum_terms / h

def caputo_approximation(x, base_activation, derivative_order, h, n):
    k = torch.arange(n).float()
    x_expanded = x.view(-1, 1)
    h_expanded = h.view(-1, 1)
    term = ((-1)**k) * scipy_gamma(derivative_order + k + 1) / (torch.tensor([math.factorial(int(k_i)) for k_i in k]) * scipy_gamma(derivative_order + 1)) * (base_activation(x_expanded - k * h_expanded) - base_activation(x_expanded - (k + 1) * h_expanded))
    sum_terms = torch.sum(term, dim=-1)
    return sum_terms / h

class CaputoFractionalActivation(nn.Module):
    def __init__(self, base_activation, derivative_order, n=10):
        super(CaputoFractionalActivation, self).__init__()
        self.base_activation = base_activation
        self.derivative_order = derivative_order
        self.n = n
        self.base_cache = {}

    def adaptive_step_size(self, x, min_step=1e-6, max_step=1e-3):
        x_normalized = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        step_size = min_step + x_normalized * (max_step - min_step)
        return step_size

    # def forward(self, x):
    #     h = self.adaptive_step_size(x)

    #     # Compute the base activation function
    #     base = torch.zeros_like(x)
    #     for i, x_i in enumerate(x.view(-1)):
    #         x_i_item = x_i.item()
    #         if torch.isnan(x_i):
    #             base.view(-1)[i] = x_i
    #             continue

    #         if x_i_item not in self.base_cache:
    #             self.base_cache[x_i_item] = self.base_activation(x_i)

    #         base.view(-1)[i] = self.base_cache[x_i_item]

    #     # Compute the Caputo approximate fractional derivative
    #     fractional_derivative = caputo_approximation(x, self.base_activation, self.derivative_order, h.view(-1), self.n)

    #     # Combine the base activation function with its fractional derivative (e.g., addition)
    #     output = base + fractional_derivative

    #     return output.view_as(x)

    def forward(self, x):
        h = self.adaptive_step_size(x)

        # Compute the base activation function
        base = torch.zeros_like(x)
        for i, x_i in enumerate(x.view(-1)):
            x_i_item = x_i.item()
            if torch.isnan(x_i):
                base.view(-1)[i] = x_i
                continue

            if x_i_item not in self.base_cache:
                self.base_cache[x_i_item] = self.base_activation(x_i)

            base.view(-1)[i] = self.base_cache[x_i_item]

        # Compute the Caputo approximate fractional derivative
        fractional_derivative = caputo_approximation(x, self.base_activation, self.derivative_order, h.view(-1), self.n)

        # Combine the base activation function with its fractional derivative (e.g., addition)
        output = base.view(-1) + fractional_derivative

        return output.view_as(x)