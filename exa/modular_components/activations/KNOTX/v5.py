import cmath
import math
import torch
import numpy as np
from scipy.integrate import solve_ivp
import torch.optim
import sympy as sp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def jones_polynomial_torus_knot(m, n):
    t = sp.symbols('t')
    numerator = t**((m-1) * (n-1)/2) * (1 - t ** (m + 1) - t**(n + 1) + t**(m+n))
    denominator = 1 - t **2
    return numerator / denominator

def convert_to_knot_representation(x):
    # Convert x to a suitable knot representation, for example, a torus knot (m, n)
    m = int(np.ceil(x))
    n = m + 1
    return (m, n)

# def knot_invariant(x):
#     knot_representation = convert_to_knot_representation(x)
#     return knot_representation
def knot_invariant(x):
    knot_representation = convert_to_knot_representation(x)
    m, n = knot_representation
    return m * n


def dynamical_systems_modeling(knot_representation):
    sigma, rho, beta = 10, 28, 8/3
    initial_state = list(knot_representation) + [0]  # Use knot_representation as the initial state
    t_span = (0, 1)
    sol = solve_ivp(lorenz_system, t_span, initial_state, args=(sigma, rho, beta), dense_output=True)
    output_value = sol.sol(1)[0]  # Get the x value at t=1
    return output_value

# def knot_gelu(x):
#     knot_inv = torch.tensor([knot_invariant(val.item()) for val in x], dtype=torch.float32, device=x.device)
#     return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * knot_inv**3)))

# def knot_gelu(x):
#     knot_inv_list = [knot_invariant(val.item()) for val in x.view(-1)]
#     knot_inv = torch.tensor(knot_inv_list, dtype=torch.float32, device=x.device).view_as(x)
#     return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * knot_inv**3)))

# def knot_gelu(x):
#     knot_inv_list = [knot_invariant(val.item()) for val in x.view(-1)]
#     knot_inv = torch.tensor(knot_inv_list, dtype=torch.float32, device=x.device).view(*x.shape)
#     return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * knot_inv**3)))

# def knot_gelu(x):
#     x_flat = x.view(-1)
#     knot_inv_list = [knot_invariant(val.item()) for val in x_flat]
#     knot_inv = torch.tensor(knot_inv_list, dtype=torch.float32, device=x.device).view_as(x_flat)
#     return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * knot_inv**3))).view_as(x)

# def knot_gelu(x):
#     x_flat = x.view(-1)
#     knot_inv_list = [knot_invariant(val.item()) for val in x_flat]
#     knot_inv = torch.tensor(knot_inv_list, dtype=torch.float32, device=x.device).view_as(x_flat)
#     return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * knot_inv**3))).view_as(x)

def knot_gelu(x):
    x_flat = x.view(-1)
    knot_inv_list = [knot_invariant(val.item()) for val in x_flat]
    knot_inv = torch.tensor(knot_inv_list, dtype=torch.float32, device=x.device).view_as(x_flat)
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * knot_inv.view_as(x)**3)))

# # Custom Activation Layer
class CustomActivation(nn.Module):
    def __init__(self, activation_type):
        super(CustomActivation, self).__init__()
        self.activation_type = activation_type

    def forward(self, x):
        if self.activation_type == 'gelu':
            return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / math.pi)) * (x + 0.044715 * torch.pow(x, 3))))
        elif self.activation_type == 'knot_gelu':
            return knot_gelu(x)
        else:
            raise ValueError("Invalid activation type")

# Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_type='gelu'):
        super(SimpleNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.activation = CustomActivation(activation_type)
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
# Generate synthetic dataset
num_samples = 1000
X = torch.randn(num_samples, input_size)
Y = torch.randint(0, output_size, (num_samples,))

# Split dataset into training and testing sets
train_ratio = 0.8
train_size = int(train_ratio * num_samples)
test_size = num_samples - train_size
X_train, X_test = torch.split(X, [train_size, test_size])
Y_train, Y_test = torch.split(Y, [train_size, test_size])



# Create DataLoaders
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# Training loop
def train(network, dataloader, optimizer, criterion, device):
    network.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(dataloader), correct / total

# Training settings
epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
nn_gelu.to(device)
nn_knot_gelu.to(device)

# Train and log progress
for epoch in range(epochs):
    gelu_loss, gelu_acc = train(nn_gelu, train_loader, optimizer_gelu, criterion, device)
    knot_gelu_loss, knot_gelu_acc = train(nn_knot_gelu, train_loader, optimizer_knot_gelu, criterion, device)

    print(f'Epoch {epoch+1}/{epochs}:')
    print(f'    GELU    | Loss: {gelu_loss:.4f} | Accuracy: {gelu_acc:.4f}')
    print(f'    KnotGELU| Loss: {knot_gelu_loss:.4f} | Accuracy: {knot_gelu_acc:.4f}')