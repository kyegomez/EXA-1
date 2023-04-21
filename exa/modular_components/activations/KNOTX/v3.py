import torch
import numpy as np
from scipy.integrate import solve_ivp
import torch.optim
import sympy as sp
import torch.nn as nn
import torch.optim as optim



def jones_polynomial_torus_knot(m, n):
    t = sp.symbols('t')
    numerator = t**((m-1) * (n-1)/2) * (1 - t ** (m + 1) - t**(n + 1) + t**(m+n))
    denominator = 1 - t **2
    return numerator / denominator



def knot_invariant(x):
    #convert the input value into a knot representation (m, n ) for a torus knot
    m, n = int(x), int(x + 1)

    #calculate the jones polynomial for the torus knot
    jones_poly = jones_polynomial_torus_knot(m, n)


    #eval the jones polynomial at a specific point
    knot_inv = jones_poly.subs('t', 2)

    return float(knot_inv)



def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def convert_to_knot_representation(x):
    # Convert x to a suitable knot representation, for example, a torus knot (m, n)
    m = int(np.ceil(x))
    n = m + 1
    return (m, n)

def knot_invariant(x):
    knot_representation = convert_to_knot_representation(x)
    return knot_representation

def dynamical_systems_modeling(knot_representation):
    sigma, rho, beta = 10, 28, 8/3
    initial_state = list(knot_representation) + [0]  # Use knot_representation as the initial state
    t_span = (0, 1)
    sol = solve_ivp(lorenz_system, t_span, initial_state, args=(sigma, rho, beta), dense_output=True)
    output_value = sol.sol(1)[0]  # Get the x value at t=1
    return output_value

def knot_gelu(x):
    knot_inv = knot_invariant(x.item())
    dyn_sys_mod = dynamical_systems_modeling(knot_inv)
    knot_gelu_output = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / np.pi)) * (x + 0.044715 * torch.pow(x, 3))) * dyn_sys_mod)
    return knot_gelu_output


#input_values = torch.tensor([-1.0, 0.0, 1.0])
#output_values = torch.tensor([knot_gelu(x) for x in input_values])
#print("Output values after applying KnotGELU activation function:", output_values)

# Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_type='gelu'):
        super(SimpleNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.activation = knot_gelu(activation_type)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

# Define input_size, hidden_size, and output_size based on your data and problem
input_size = 10
hidden_size = 20
output_size = 2

# Initialize Simple Neural Networks with GELU and KnotGELU activations
nn_gelu = SimpleNN(input_size, hidden_size, output_size, activation_type='gelu')
nn_knot_gelu = SimpleNN(input_size, hidden_size, output_size, activation_type='knot_gelu')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_gelu = optim.SGD(nn_gelu.parameters(), lr=0.01)
optimizer_knot_gelu = optim.SGD(nn_knot_gelu.parameters(), lr=0.01)

# Train the networks and compare their performance on your dataset