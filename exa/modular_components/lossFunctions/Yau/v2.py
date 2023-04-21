import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances


def geometric_similarity(y_pred, y_true):
    # Compute a simple geometric metric based on the L2 norm of the difference between y_pred and y_true
    geometric_difference = np.linalg.norm(y_pred - y_true)
    return geometric_difference


# Helper function to perturb the input data
def perturb_input_data(x, perturbation):
    # Apply the perturbation to the input data and return the perturbed data
    return x + perturbation

# Generic topological invariance function
def topological_invariance(y_pred, y_true, metric='euclidean'):
    distance_matrix = pairwise_distances(y_pred, y_true, metric=metric)
    distance = np.sum(np.min(distance_matrix, axis=1))  # Use the sum of minimum distances as the discrepancy metric
    return distance

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