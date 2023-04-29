# from nebulaV4 import one_hot_encoding
# from nebulaV4 import Nebula
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn as nn

# class LossFunction:
#     def compute_loss(self, y_pred, y_true):
#         raise NotImplemented("compute_loss method must be implemented")
    

# #implement specific loss functions that inherit from LossFunction
# class L1Loss(LossFunction):
#     def __init__(self):
#         self.loss_function = nn.L1Loss()

#     def compute_loss(self, y_pred, y_true):
#         return self.loss_function(y_pred, y_true)
    

# class MSELoss(LossFunction):
#     def __init__(self):
#         self.loss_function = nn.MSELoss()

#     def compute_loss(self, y_pred, y_true):
#         return self.loss_function(y_pred, y_true)
    

# class CrossEntropyLoss(LossFunction):
#     def __init__(self):
#         self.loss_function = nn.CrossEntropyLoss()

#     def compute_loss(self, y_pred, y_true):
#         return self.loss_function(y_pred, y_true)
    
# def prepare_targets(loss_function, y_true, num_classes=None):
#     if isinstance(loss_function, L1Loss) and num_classes is not None:
#         return one_hot_encoding(y_true, num_classes)
#     return y_true

# def generate_classification_data(num_samples, num_classes):
#     y_true = torch.randint(0, num_classes, (num_samples,))
#     y_pred = torch.rand(num_samples, num_classes)
#     return y_pred, y_true

# def generate_regression_data(num_samples):
#     y_true = torch.randn(num_samples)
#     y_pred = torch.randn(num_samples)
#     return y_pred, y_true

# def test_loss_functions(loss_functions, y_pred, y_true, num_classes=None):
#     results = []
#     for loss_function in loss_functions:
#         prepared_y_true = prepare_targets(loss_function, y_true, num_classes)
#         loss = loss_function.compute_loss(y_pred, prepared_y_true)
#         results.append(loss.item())
#     return results

# def plot_loss_comparison(loss_functions, losses):
#     loss_function_names = [loss_function.__class__.__name__ for loss_function in loss_functions]
#     plt.bar(loss_function_names, losses)
#     plt.xlabel("Loss Functions")
#     plt.ylabel("Loss Value")
#     plt.title("Loss Function Comparison")
#     plt.show()

# batch_size = 100
# num_classes = 5
# y_true_classification = torch.randint(0, num_classes, (batch_size,))
# num_classes = y_true_classification.max().item() + 1

# # Generate classification data
# y_pred_classification, y_true_classification = generate_classification_data(num_classes, num_classes)

# # Generate regression data
# y_pred_regression, y_true_regression = generate_regression_data(num_classes)

# # Loss functions to compare
# loss_functions = [Nebula(), L1Loss(), MSELoss(), CrossEntropyLoss()]

# # Test classification data
# print("Classification Losses:")
# classification_losses = test_loss_functions(loss_functions, y_pred_classification, y_true_classification, num_classes=num_classes)

# # Test regression data
# print("\nRegression Losses:")
# regression_losses = test_loss_functions(loss_functions, y_pred_regression, y_true_regression)

# # Plot comparison
# print("\nLoss Comparison for Classification:")
# plot_loss_comparison(loss_functions, classification_losses)

# print("\nLoss Comparison for Regression:")
# plot_loss_comparison(loss_functions, regression_losses)

# from nebulaV4 import one_hot_encoding
# from nebulaV4 import Nebula
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from nebulav2 import Nebula
from nebulav2 import one_hot_encoding
import torch.nn.functional as F


def generate_multilabel_classification_data(num_samples, num_classes):
    y_true = torch.randint(0, 2, (num_samples, num_classes)).float()
    y_pred = torch.rand(num_samples, num_classes)
    return y_pred, y_true



class LossFunction:
    def compute_loss(self, y_pred, y_true):
        raise NotImplemented("compute_loss method must be implemented")

class L1Loss(LossFunction):
    def __init__(self):
        self.loss_function = nn.L1Loss()

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)

class MSELoss(LossFunction):
    def __init__(self):
        self.loss_function = nn.MSELoss()

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)

class CrossEntropyLoss(LossFunction):
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss()

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)
    


class MultiLabelSoftMarginLoss(LossFunction):
    def __init__(self):
        self.loss_function = nn.MultiLabelSoftMarginLoss()

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)
    

class PoissonNLLoss(LossFunction):
    def __init__(self):
        self.loss_function = nn.PoissonNLLLoss()

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)
    
class KLDivLoss(LossFunction):
    def __init__(self):
        self.loss_function = nn.KLDivLoss()

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(F.log_softmax(y_pred, dim=1), y_true)
    
class PoissonNLLLoss(LossFunction):
    def __init__(self):
        self.loss_function = nn.PoissonNLLLoss()

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)

# def prepare_targets(loss_function, y_true, num_classes=None):
#     if isinstance(loss_function, L1Loss) and num_classes is not None:
#         return one_hot_encoding(y_true, num_classes)
#     return y_true
def prepare_targets(loss_function, y_true, num_classes=None):
    if (isinstance(loss_function, L1Loss) or isinstance(loss_function, MSELoss)) and num_classes is not None:
        return one_hot_encoding(y_true, num_classes)
    if isinstance(loss_function, PoissonNLLLoss):
        return y_true.view(-1, 1).expand(-1, num_classes)
    if isinstance(loss_function, KLDivLoss):
        return y_true.float()
    return y_true

def generate_classification_data(num_samples, num_classes, for_poisson_nll=False):
    y_true = torch.randint(0, num_classes, (num_samples,))
    if for_poisson_nll:
        y_true = y_true.view(-1, 1).expand(-1, num_classes).float()
    y_pred = torch.rand(num_samples, num_classes)
    return y_pred, y_true


def generate_regression_data(num_samples):
    y_true = torch.abs(torch.randn(num_samples))
    y_pred = torch.randn(num_samples)
    return y_pred, y_true

# def test_loss_functions(loss_functions, y_pred, y_true, num_classes=None):
#     results = []
#     for loss_function in loss_functions:
#         prepared_y_true = prepare_targets(loss_function, y_true, num_classes)
#         loss = loss_function.compute_loss(y_pred, prepared_y_true)
#         results.append(loss.item())
#     return results

def test_loss_functions(loss_functions, y_pred, y_true, num_classes=None):
    losses = []
    for loss_function in loss_functions:
        for_poisson_nll = isinstance(loss_function, PoissonNLLLoss)
        if num_classes is not None and not for_poisson_nll:
            y_true = y_true.squeeze()
        elif for_poisson_nll:
            y_true = y_true.view(-1, 1).expand(-1, num_classes)
        prepared_y_true = prepare_targets(loss_function, y_true, num_classes)
        loss = loss_function.compute_loss(y_pred, prepared_y_true)
        losses.append(loss.item())
    return losses

def plot_loss_comparison(loss_functions, losses):
    loss_function_names = [loss_function.__class__.__name__ for loss_function in loss_functions]
    plt.bar(loss_function_names, losses)
    plt.xlabel("Loss Functions")
    plt.ylabel("Loss Value")
    plt.title("Loss Function Comparison")
    plt.show()

batch_size = 100
num_classes = 5
y_true_classification = torch.randint(0, num_classes, (batch_size,))
num_classes = y_true_classification.max().item() + 1

# Generate classification data
y_pred_classification, y_true_classification = generate_classification_data(batch_size, num_classes)
y_pred_multilabel_classification, y_true_multilabel_classification = generate_multilabel_classification_data(batch_size, num_classes)


# Generate regression data
y_pred_regression, y_true_regression = generate_regression_data(batch_size)

# Loss functions to compare
loss_functions = [Nebula(), L1Loss(), MSELoss(), CrossEntropyLoss(), PoissonNLLoss(), KLDivLoss(), PoissonNLLLoss()]

# Test classification data
# # Test classification data
print("Classification Losses:")
classification_losses = test_loss_functions(loss_functions, y_pred_classification, y_true_classification, num_classes=num_classes)

# Test regression data
print("\nRegression Losses:")
regression_losses = test_loss_functions(loss_functions, y_pred_regression, y_true_regression)

# Plot comparison
print("\nLoss Comparison for Classification:")
plot_loss_comparison(loss_functions, classification_losses)

print("\nLoss Comparison for Regression:")
plot_loss_comparison(loss_functions, regression_losses)

# Test multi-label classification data
print("Multi-label Classification Losses:")
multilabel_classification_losses = test_loss_functions(loss_functions, y_pred_multilabel_classification, y_true_multilabel_classification, num_classes=num_classes)

# Plot comparison
print("\nLoss Comparison for Multi-label Classification:")
plot_loss_comparison(loss_functions, multilabel_classification_losses)