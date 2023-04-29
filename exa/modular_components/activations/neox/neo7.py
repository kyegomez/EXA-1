import torch
import torch.nn as nn
import math
import torch.jit

def relu_activation(x):
    return torch.relu(x)

@torch.jit.script
def caputo_approximation(x, derivative_order, h, n):
    k = torch.arange(n).float().view(1, -1)
    x_expanded = x.view(-1, 1)
    h_expanded = h.view(-1, 1)
    
    factorial_k = torch.empty_like(k)
    for i in range(k.shape[1]):
        factorial_k[0, i] = math.factorial(int(k[0, i]))
    
    term = ((-1)**k) * torch.exp(torch.lgamma(derivative_order + k + 1)) / (factorial_k * torch.exp(torch.lgamma(derivative_order + 1))) * (relu_activation(x_expanded - k * h_expanded) - relu_activation(x_expanded - (k + 1) * h_expanded))
    sum_terms = torch.sum(term, dim=-1)
    return sum_terms / h

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class CaputoFractionalActivation(nn.Module):
    def __init__(self, base_activation, derivative_order, n=20):
        super(CaputoFractionalActivation, self).__init__()
        self.base_activation = base_activation
        self.derivative_order = torch.tensor(float(derivative_order))
        self.n = torch.tensor(n)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def adaptive_step_size(self, x, min_step=1e-6, max_step=1e-3):
        x_mean = torch.mean(x)
        x_std = torch.std(x)
        step_size = min_step + self.alpha * (x - x_mean) / x_std * (max_step - min_step)
        return step_size

    def forward(self, x):
        h = self.adaptive_step_size(x)
        base = self.base_activation(x)
        fractional_derivative = caputo_approximation(x, self.derivative_order, h.view(-1), self.n)
        output = base.view_as(fractional_derivative) + fractional_derivative
        return output.view_as(x)