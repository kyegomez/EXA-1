import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
import torch
import numpy as np
import networkx as nx
from gudhi import SimplexTree
import gudhi as gd


def geometric_similarity(y_pred, y_true):
    # Compute a simple geometric metric based on the L2 norm of the difference between y_pred and y_true
    geometric_difference = np.linalg.norm(y_pred - y_true)
    return geometric_difference


# Helper function to perturb the input data
def perturb_input_data(x, perturbation):
    # Apply the perturbation to the input data and return the perturbed data
    return x + perturbation

# Generic topological invariance function
# def topological_invariance(y_pred, y_true, metric='euclidean'):
#     distance_matrix = pairwise_distances(y_pred, y_true, metric=metric)
#     distance = np.sum(np.min(distance_matrix, axis=1))  # Use the sum of minimum distances as the discrepancy metric
#     return distance



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


# def bottleneck_distance(distance_matrix_1, distance_matrix_2):
#     # Step 1: Compute the persistence diagrams for both distance matrices
#     rips_complex_1 = gd.RipsComplex(distance_matrix=distance_matrix_1, max_edge_length=np.inf)
#     simplex_tree_1 = rips_complex_1.create_simplex_tree(max_dimension=2)
#     persistence_diagram_1 = simplex_tree_1.persistence()

#     rips_complex_2 = gd.RipsComplex(distance_matrix=distance_matrix_2, max_edge_length=np.inf)
#     simplex_tree_2 = rips_complex_2.create_simplex_tree(max_dimension=2)
#     persistence_diagram_2 = simplex_tree_2.persistence()

#     # Step 2: Calculate the bottleneck distance between the two persistence diagrams
#     bottleneck_distance_value = gd.bottleneck_distance(persistence_diagram_1, persistence_diagram_2)

#     # Step 3: Return the bottleneck distance
#     return bottleneck_distance_value


#v2
def bottleneck_distance(distance_matrix_1, distance_matrix_2):
    # Step 1: Compute the persistence diagrams for both distance matrices
    rips_complex_1 = gd.RipsComplex(distance_matrix=distance_matrix_1, max_edge_length=np.inf)
    simplex_tree_1 = rips_complex_1.create_simplex_tree(max_dimension=2)
    persistence_diagram_1 = np.array(simplex_tree_1.persistence())

    rips_complex_2 = gd.RipsComplex(distance_matrix=distance_matrix_2, max_edge_length=np.inf)
    simplex_tree_2 = rips_complex_2.create_simplex_tree(max_dimension=2)
    persistence_diagram_2 = np.array(simplex_tree_2.persistence())

    # Step 2: Calculate the bottleneck distance between the two persistence diagrams
    bottleneck_distance_value = gd.bottleneck_distance(persistence_diagram_1, persistence_diagram_2)

    # Step 3: Return the bottleneck distance
    return bottleneck_distance_value



def infer_data_type(y_pred, y_true):
    if y_pred.ndim() == 2 and y_true.ndim() == 2:
        return 'point_cloud'
    elif isinstance(y_pred, nx.Graph) and isinstance(y_true, nx.Graph):
        return 'graph'
    elif isinstance(y_pred, list) and isinstance(y_true, list):
        return 'multi_modal'
    else:
        raise ValueError('Unsupported data type.')


# def infer_data_type(y_pred, y_true):
    # Analyze the input data (y_pred and y_true) and determine the data type
    # This function should return a string representing the data type, such as 'point_cloud', 'graph', or 'multi_modal'

    # if y_pred.ndim == 2 and y_true.ndim == 2:
    #     # If both y_pred and y_true are 2D arrays, we can assume they represent point clouds or graphs
    #     if is_point_cloud(y_pred, y_true):
    #         return 'point_cloud'
    #     elif is_graph(y_pred, y_true):
    #         return 'graph'
    # elif y_pred.ndim == 3 and y_true.ndim == 3:
    #     # If both y_pred and y_true are 3D arrays, we can assume they represent multi-modal data
    #     if is_multi_modal(y_pred, y_true):
    #         return 'multi_modal'

    # raise ValueError("Unable to infer data type.")

#v3
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

    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)

    # Calculate the pairwise distance matrices for both point clouds
    y_pred_distance_matrix = torch.cdist(y_pred, y_pred)

    #v1
    # y_true_distance_matrix = torch.cdist(y_true, y_true)

    #v2
    y_true_distance_matrix = torch.cdist(y_true.float(), y_true.float())


    # Calculate the topological invariance metric, e.g., bottleneck distance
    # You'll need to implement the `bottleneck_distance` function or use an existing implementation
    topological_invariance_metric = bottleneck_distance(y_pred_distance_matrix.numpy(), y_true_distance_matrix.numpy())

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

# Note: The `bottleneck_distance` function mentioned in the point_cloud_topological_invariance function is not provided
# here. You'll need to either implement it yourself or use an existing implementation from a library like GUDHI.









def complexity_reduction(network):
    # Example: Compute the L1 regularization term for the network's weights
    l1_regularization = 0
    for layer in network.layers:
        if hasattr(layer, 'kernel'):  # Check if the layer has trainable weights
            l1_regularization += np.sum(np.abs(layer.kernel))
    return l1_regularization

# Revised stability function
def stability(model, x, y_true, perturbations):
    stability_metric = 0
    for perturbation in perturbations:
        x_perturbed = perturb_input_data(x, perturbation)
        y_pred_perturbed = model.predict(x_perturbed)
        stability_metric += np.linalg.norm(y_pred_perturbed - y_true)
    stability_metric /= len(perturbations)
    return stability_metric

# Calabi-Yau inspired loss function
# def calabi_yau_loss(model, x, y_true, perturbations, alpha=0.1, beta=0.1, gamma=0.1):
#     y_pred = model.predict(x)

#     # Compute geometric similarity
#     geom_sim = geometric_similarity(y_pred, y_true)

#     # Compute topological invariance
#     topo_inv = topological_invariance(y_pred, y_true)

#     # Compute complexity reduction
#     comp_red = complexity_reduction(model)

#     # Compute stability
#     stab = stability(model, x, y_true, perturbations)

#     # Combine the components with weighting factors alpha, beta, and gamma
#     total_loss = geom_sim + alpha * topo_inv + beta * comp_red + gamma * stab

#     return total_loss


#v2
#reshape arrays to 2d
def calabi_yau_loss(model, x, y_true, perturbations, alpha=0.1, beta=0.1, gamma=0.1):
    y_pred = model.predict(x)
    
    # Reshape y_pred and y_true to 2D arrays
    y_pred = y_pred.reshape(-1, 1)
    y_true = y_true.reshape(-1, 1)

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