import torch
import torch.nn as nn

def fractional_derivative(x, base_activation, derivative_order, h):
    base = base_activation(x)
    base_plus_h = base_activation(x + h)
    fractional_derivative = ((base_plus_h - base) / h) ** derivative_order
    return fractional_derivative

class SimplifiedOptimizedFractionalActivation(nn.Module):
    def __init__(self, base_activation, derivative_order):
        super(SimplifiedOptimizedFractionalActivation, self).__init__()
        self.base_activation = base_activation
        self.derivative_order = derivative_order
        self.base_cache = {}

    # def adaptive_step_size(x, min_step=1e-6, max_step=1e-3):
    #    # Normalize the input x
    #     x_normalized = (x - x.min()) / (x.max() - x.min())

    #    # Calculate the desired step size based on the normalized input
    #     step_size = min_step + x_normalized * (max_step - min_step)

    #     return step_size

    #v2 TypeError: min(): argument 'input' (position 1) must be Tensor, not SimplifiedOptimizedFractionalActivation
    # def adaptive_step_size(x, min_step=1e-6, max_step=1e-3):
    #     # Normalize the input x
    #     x_normalized = (x - torch.min(x)) / (torch.max(x) - torch.min(x))

    #     # Calculate the desired step size based on the normalized input
    #     step_size = min_step + x_normalized * (max_step - min_step)

    #     return step_size

    #v3
    def adaptive_step_size(self, x, min_step=1e-6, max_step=1e-3):
        # Normalize the input x
        x_normalized = (x - torch.min(x)) / (torch.max(x) - torch.min(x))

        # Calculate the desired step size based on the normalized input
        step_size = min_step + x_normalized * (max_step - min_step)

        return step_size

   
   #v1 -RuntimeError: a Tensor with 8192 elements cannot be converted to Scalar

    # def forward(self, x):
    #     h = self.adaptive_step_size(x)

    #     # Caching base activation
    #     if x.item() not in self.base_cache:
    #         self.base_cache[x.item()] = self.base_activation(x)

    #     base = self.base_cache[x.item()]

    #     # Approximate fractional derivative
    #     fractional_derivative = fractional_derivative(x, self.base_activation, self.derivative_order, h)

    #     # Combine the base activation function with its fractional derivative (e.g., addition)
    #     output = base + fractional_derivative

    #     return output

    #v2 -> UnboundLocalError: cannot access local variable 'fractional_derivative' where it is not associated with a value
    # def forward(self, x):
    #     h = self.adaptive_step_size(x)

    #     # Caching base activation
    #     output = torch.zeros_like(x)
    #     for i, x_i in enumerate(x.view(-1)):
    #         if x_i.item() not in self.base_cache:
    #             self.base_cache[x_i.item()] = self.base_activation(x_i)

    #         base = self.base_cache[x_i.item()]

    #         # Approximate fractional derivative
    #         fractional_derivative = fractional_derivative(x_i, self.base_activation, self.derivative_order, h[i])

    #         # Combine the base activation function with its fractional derivative (e.g., addition)
    #         output[i] = base + fractional_derivative

    #     return output.view_as(x)

    #v3 
    # def forward(self, x):
    #     h = self.adaptive_step_size(x)

    #     # Caching base activation
    #     output = torch.zeros_like(x)
    #     for i, x_i in enumerate(x.view(-1)):
    #         if x_i.item() not in self.base_cache:
    #             self.base_cache[x_i.item()] = self.base_activation(x_i)

    #         base = self.base_cache[x_i.item()]

    #         # Approximate fractional derivative
    #         frac_derivative = fractional_derivative(x_i, self.base_activation, self.derivative_order, h[i])

    #         # Combine the base activation function with its fractional derivative (e.g., addition)
    #         output[i] = base + frac_derivative

    #     return output.view_as(x)

    #v4 -> IndexError: index 64 is out of bounds for dimension 0 with size 64
    #IndexError: index 64 is out of bounds for dimension 0 with size 64
    # def forward(self, x):
    #     h = self.adaptive_step_size(x)

    #     # Caching base activation
    #     output = torch.zeros_like(x)
    #     for i, x_i in enumerate(x.view(-1)):
    #         if x_i.item() not in self.base_cache:
    #             self.base_cache[x_i.item()] = self.base_activation(x_i)

    #         base = self.base_cache[x_i.item()]

    #         # Approximate fractional derivative
    #         frac_derivative = fractional_derivative(x_i, self.base_activation, self.derivative_order, h.view(-1)[i])

    #         # Combine the base activation function with its fractional derivative (e.g., addition)
    #         output[i] = base + frac_derivative

    #     return output.view_as(x)

    #v5 - KeyError: nan
    def forward(self, x):
        h = self.adaptive_step_size(x)

        # Caching base activation
        output = torch.zeros_like(x)
        for i, x_i in enumerate(x.view(-1)):
            if x_i.item() not in self.base_cache:
                self.base_cache[x_i.item()] = self.base_activation(x_i)

            base = self.base_cache[x_i.item()]

            # Approximate fractional derivative
            frac_derivative = fractional_derivative(x_i, self.base_activation, self.derivative_order, h.view(-1)[i])

            # Combine the base activation function with its fractional derivative (e.g., addition)
            output.view(-1)[i] = base + frac_derivative

        return output.view_as(x)
    
    #v6 
    def forward(self, x):
        h = self.adaptive_step_size(x)

        # Caching base activation
        output = torch.zeros_like(x)
        for i, x_i in enumerate(x.view(-1)):
            x_i_item = x_i.item()
            if torch.isnan(x_i):
                output.view(-1)[i] = x_i
                continue

            if x_i_item not in self.base_cache:
                self.base_cache[x_i_item] = self.base_activation(x_i)

            base = self.base_cache[x_i_item]

            # Approximate fractional derivative
            frac_derivative = fractional_derivative(x_i, self.base_activation, self.derivative_order, h.view(-1)[i])

            # Combine the base activation function with its fractional derivative (e.g., addition)
            output.view(-1)[i] = base + frac_derivative

        return output.view_as(x)