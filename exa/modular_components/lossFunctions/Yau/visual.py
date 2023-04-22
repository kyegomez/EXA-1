
import torch
import numpy as np
import networkx as nx
from gudhi import SimplexTree
import gudhi as gd


def geometric_similarity(y_pred, y_true):
    # Compute a simple geometric metric based on the L2 norm of the difference between y_pred and y_true
    geometric_difference = torch.norm(y_pred - y_true)
    return geometric_difference


# Helper function to perturb the input data
def perturb_input_data(x, perturbation):
    # Apply the perturbation to the input data and return the perturbed data
    return x + perturbation


def topological_invariance(y_pred, y_true, **kwargs):
    data_type = infer_data_type(y_pred, y_true)

    if data_type == 'point_cloud':
        return point_cloud_topological_invariance(y_pred, y_true, **kwargs)
    elif data_type == 'graph':
        return graph_topological_invariance(y_pred, y_true, **kwargs)
    elif data_type == 'multi_modal':
        return multi_modal_topological_invariance(y_pred, y_true, **kwargs)
    else:
        raise ValueError(f'Unsupported data type: {data_type}')


def infer_data_type(y_pred, y_true):
    if y_pred.ndim == 2 and y_true.ndim == 2:
        return 'point_cloud'
    elif isinstance(y_pred, nx.Graph) and isinstance(y_true, nx.Graph):
        return 'graph'
    elif isinstance(y_pred, list) and isinstance(y_true, list):
        return 'multi_modal'
    else:
        raise ValueError('Unsupported data type.')


def point_cloud_topological_invariance(y_pred, y_true, **kwargs):
    # Calculate the pairwise distance matrices for both point clouds
    y_pred_distance_matrix = torch.cdist(y_pred, y_pred)
    y_true_distance_matrix = torch.cdist(y_true.float(), y_true.float())

    # Calculate the topological invariance metric, e.g., bottleneck distance
    # topological_invariance_metric = bottleneck_distance(y_pred_distance_matrix.numpy(), y_true_distance_matrix.numpy())
    topological_invariance_metric = bottleneck_distance(y_pred_distance_matrix.detach().numpy(), y_true_distance_matrix.detach().numpy())

    return topological_invariance_metric


def graph_topological_invariance(y_pred, y_true, **kwargs):
    # Calculate the graph edit distance between the predicted and true graphs
    # You can use the NetworkX library for this
    graph_edit_distance = nx.graph_edit_distance(y_pred, y_true)

    return graph_edit_distance


def multi_modal_topological_invariance(y_pred, y_true, **kwargs):
    # Calculate the topological invariance metric for multi-modal data
    # This could be a combination of different topological invariance metrics
    # based on the specific problem requirements

    # Example: Calculate the topological invariance for each modality and average the results
    num_modalities = len(y_pred)
    total_topological_invariance = 0.0

    for i in range(num_modalities):
        data_type = infer_data_type(y_pred[i], y_true[i])
        topological_invariance_i = topological_invariance(y_pred[i], y_true[i], data_type=data_type)
        total_topological_invariance += topological_invariance_i

    average_topological_invariance = total_topological_invariance / num_modalities

    return average_topological_invariance


def bottleneck_distance(distance_matrix_1, distance_matrix_2):
    # Step 1: Compute the persistence diagrams for both distance matrices
    rips_complex_1 = gd.RipsComplex(distance_matrix=distance_matrix_1, max_edge_length=np.inf)
    simplex_tree_1 = rips_complex_1.create_simplex_tree(max_dimension=2)
    # persistence_diagram_1 = np.array(simplex_tree_1.persistence())
    persistence_diagram_1 = np.array([pair[1] for pair in simplex_tree_1.persistence()])


    rips_complex_2 = gd.RipsComplex(distance_matrix=distance_matrix_2, max_edge_length=np.inf)
    simplex_tree_2 = rips_complex_2.create_simplex_tree(max_dimension=2)
    persistence_diagram_2 = np.array([pair[1] for pair in simplex_tree_2.persistence()])
    # persistence_diagram_2 = np.array(simplex_tree_2.persistence())

    # Step 2: Calculate the bottleneck distance between the two persistence diagrams
    bottleneck_distance_value = gd.bottleneck_distance(persistence_diagram_1, persistence_diagram_2)

    # Step 3: Return the bottleneck distance
    return bottleneck_distance_value

# Generic function for complexity reduction
def complexity_reduction(model):
    # Example: Compute the L1 regularization term for the network's weights
    l1_regularization = 0
    for parameter in model.parameters():
        l1_regularization += torch.sum(torch.abs(parameter))
    return l1_regularization


# Revised stability function
def stability(model, x, y_true, perturbations):
    stability_metric = 0
    for perturbation in perturbations:
        x_perturbed = perturb_input_data(x, perturbation)
        y_pred_perturbed = model(x_perturbed)
        stability_metric += torch.norm(y_pred_perturbed - y_true)
    stability_metric /= len(perturbations)
    return stability_metric


# Calabi-Yau inspired loss function
def calabi_yau_loss(model, x, y_true, perturbations, alpha=0.1, beta=0.1, gamma=0.1):

    # y_pred = torch.Tensor(y_pred.numpy()).view(-1, 1)

    y_pred = model(x)

    # Reshape y_pred and y_true to 2D tensors
    y_pred = y_pred.view(-1, 1)
    y_true = y_true.view(-1, 1)

    # Compute geometric similarity
    geom_sim = geometric_similarity(y_pred, y_true)

    # Compute topological invariance
    topo_inv = topological_invariance(y_pred, y_true)

    # Compute complexity reduction
    comp_red = complexity_reduction(model)

    # Compute stability
    stab = stability(model, x, y_true, perturbations)

    # Combine the components with weighting factors alpha, beta, and gamma
    total_loss = geom_sim + alpha * topo_inv + beta * comp_red + gamma * stab

    return total_loss

import torch.nn as nn

# Define a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

import torch
import matplotlib.pyplot as plt
import numpy as np

# Create an instance of the linear regression model
model = LinearRegression()

# Compute the loss for a given input value
x = torch.Tensor([1])
y_true = torch.Tensor([2])
perturbations = torch.Tensor([[0.1], [-0.1]])
alpha = 0.1
beta = 0.1
gamma = 0.1

loss = calabi_yau_loss(model, x, y_true, perturbations, alpha=alpha, beta=beta, gamma=gamma)

# Define the range of values for the independent variable (x-axis)
x_range = np.linspace(-10, 10, 100)

# Set the hyperparameters for the loss function
alpha = 0.1
beta = 0.1
gamma = 0.1

# Define the true output values for the input values in the x range
y_true = np.sin(x_range)

# Convert y_true to a tensor
y_true_tensor = torch.Tensor(y_true)

# Define the perturbations to use for computing the stability component of the loss function
perturbations = np.random.normal(0, 0.1, size=(10, len(x_range)))

# Initialize an empty list to store the loss values for each input value in the x range
loss_values = []

# Loop through each input value in the x range and compute the corresponding loss value
for x in x_range:
    x_tensor = torch.Tensor([x])
    y_pred = torch.Tensor([np.sin(x)])
    loss = calabi_yau_loss(model, x_tensor, y_true_tensor, perturbations, alpha=alpha, beta=beta, gamma=gamma)
    loss_values.append(loss.item())

# Plot the loss values against the input values
plt.plot(x_range, loss_values)
plt.xlabel('Input Values')
plt.ylabel('Loss')
plt.title('Calabi-Yau Loss')
plt.show()