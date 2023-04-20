import numpy as np
from torch.nn import BCELoss

# Define the base LossFunction class
class LossFunction:
    def compute_loss(self, y_pred, y_true):
        raise NotImplementedError("compute_loss method must be implemented in the derived class")

# Define specific loss function classes that inherit from LossFunction
class CrossEntropyLoss(LossFunction):
    def compute_loss(self, y_pred, y_true):
        softmax_pred = self.softmax(y_pred)
        loss = -np.sum(y_true * np.log(softmax_pred))
        return loss

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class MeanSquaredErrorLoss(LossFunction):
    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)


#v1 

# # Create a DynamicLossFunction class that inherits from the LossFunction base class
# class DynamicLossFunction(LossFunction):
#     def __init__(self):
#         self.loss_function = None

#     def determine_loss_function(self, y_pred, y_true):
#         # Implement logic to determine the type of data and select the appropriate loss function
#         # For example, based on the shape of y_true or other criteria
#         if y_true.shape[1] > 1:
#             self.loss_function = CrossEntropyLoss()
#         else:
#             self.loss_function = MeanSquaredErrorLoss()

#     def compute_loss(self, y_pred, y_true):
#         self.determine_loss_function(y_pred, y_true)
#         return self.loss_function.compute_loss(y_pred, y_true)

class DynamicLossFunction(LossFunction):
    def __init__(self):
        self.loss_function = None

    def determine_loss_function(self, y_pred, y_true):
        # Implement logic to determine the type of data and select the appropriate loss function
        # Check if the problem is a classification or regression task
        is_classification = self.is_classification_task(y_true)

        # Check if the problem involves multiple classes or binary classes
        is_multiclass = self.is_multiclass_problem(y_true)

        # Select the appropriate loss function based on the problem type
        if is_classification:
            if is_multiclass:
                self.loss_function = CrossEntropyLoss()
            else:
                self.loss_function = BCELoss()
        else:
            self.loss_function = MeanSquaredErrorLoss()

    def is_classification_task(self, y_true):
        # Check if the target variable is binary or consists of integers (indicating class labels)
        return np.issubdtype(y_true.dtype, np.integer)

    def is_multiclass_problem(self, y_true):
        # Check if the problem involves multiple classes by counting the unique values in y_true
        unique_values = np.unique(y_true)
        return len(unique_values) > 2

    def compute_loss(self, y_pred, y_true):
        self.determine_loss_function(y_pred, y_true)
        return self.loss_function.compute_loss(y_pred, y_true)

# Example usage
y_pred_classification = np.array([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]])
y_true_classification = np.array([[1, 0, 0], [0, 1, 0]])

y_pred_regression = np.array([[2.5], [3.2]])
y_true_regression = np.array([[2.0], [3.0]])

dynamic_loss_function = DynamicLossFunction()

loss_classification = dynamic_loss_function.compute_loss(y_pred_classification, y_true_classification)
print("Dynamic loss for classification:", loss_classification)

loss_regression = dynamic_loss_function.compute_loss(y_pred_regression, y_true_regression)
print("Dynamic loss for regression:", loss_regression)