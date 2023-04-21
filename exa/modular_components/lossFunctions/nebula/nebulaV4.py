import torch 
import torch.nn as nn
import numpy as np


#define the loss function class
class LossFunction:
    def compute_loss(self, y_pred, y_true):
        raise NotImplemented("compute_loss method must be implemented")
    

#implement specific loss functions that inherit from LossFunction
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
    

"""

all pytorch loss functions

"""


#v1
# class Nebula(LossFunction):
#     def __init__(self):
#         self.loss_function = None
    
#     def determine_loss_function(self, y_pred, y_true):
#         ##implement logic to determine type of data and select the loss function
#         #based on the shape of y_true or other criteria
#         if len(y_true.shape) > 1 and y_true.shape[1] > 1:
#             self.loss_function = CrossEntropyLoss()
#         else:
#             self.loss_function = MSELoss()

#     #transform function data1 to -> data type loss function can understand?

#     def compute_loss(self, y_pred, y_true):
#         self.determine_loss_function(y_pred, y_true)
#         return self.loss_function.compute_loss(y_pred, y_true)
    
# Example usage
# y_pred_classification = torch.tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]])
# y_true_classification = torch.tensor([0, 1])



#v2
# GRADIENT BOOSTED 
# greedy algorithm
#concurrency
#asynchrony
#CACHING FOR TRAINING --> THIS IS YOUR DATASET -> OKAY HERES LOSS FUNCTION -> DONT COMPUTE DETERMINE LOSS FUNCTION
#self healing loss function
#responsive loss function
# 1 loss function for any task


def one_hot_encoding(y_true, num_classes):
    y_true_one_hot = torch.zeros(y_true.size(0), num_classes)
    y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)
    return y_true_one_hot

class Nebula(LossFunction):
    def __init__(self, domain_knowledge=None, user_input=None):
        self.loss_function = None
        self.domain_knowledge = domain_knowledge
        self.user_input = user_input
        self.loss_function_cache = {}


    def determine_loss_function(self, y_pred, y_true):
        is_classification = None

        #op 1 check range of values in y_true
        if is_classification is None:
            unique_values = np.unique(y_true)
            if len(unique_values) <= 10 and np.all(np.equal(np.mod(unique_values, 1), 0)): 
                is_classification = True

        #==============================================>

        

        #opt2 - check the distribution of valus in y_true
        # assuming a specific pattern indicates a classification problem
        # You can replace this with a more specific pattern check if needded
        # value_counts = np.bincount(y_true.flatten().astype(int))
        if is_classification is None:
            value_counts = np.bincount(y_true.flatten().to(dtype=torch.int32).numpy())
            if np.all(value_counts > 0):
                is_classification = True
        

        #==============================================>


        #op 3 analyze the dimension of y_pred
        if y_pred.ndim > 2:
             #handle segmentation problem
          pass

        #==============================================>


        #op4 -> check sparsity of y_true
        #v1
        # sparsity = np.count_nonzero(y_true) / y_true.numel()
        # if sparsity < 0.1:
        #     #handle multi label classification problem
        #     pass

        #v2
        # sparsity = np.count_nonzero(y_true) / y_true.numel()
        # if sparsity < 0.5:
        #     self.loss_function = torch.nn.BCEWithLogitsLoss()

        #v3
        # sparsity = np.count_nonzero(y_true) / y_true.numel()
        # if sparsity < 0.5:
        #     self.loss_function = torch.nn.BCEWithLogitsLoss()
        #     self.compute_loss = self.loss_function

        if is_classification is None:
            sparsity = np.count_nonzero(y_true) / y_true.numel()
            if sparsity < 0.5:
                self.loss_functions = torch.nn.BCEWithLogitsLoss()
                self.compute_Loss = self.loss_function
                return

        #==============================================>


        #op5 analyze the relationship between y_pred and y_true
        #v1
        # correlation = np.corrcoef(y_pred.flatten(), y_true.flatten())[0, 1]
        # if correlation > 0.8:
        #     is_classification = False

        #v2 
        # y_pred_flat = y_pred.flatten().numpy()
        # y_true_flat = y_true.flatten().numpy()
        # if y_pred_flat.shape != y_true_flat.shape:
        #     y_pred_flat = y_pred_flat[:y_true_flat.shape]
        # correlation = np.corrcoef(y_pred_flat, y_true_flat)[0, 1]

        #v3 
        # y_pred_flat = y_pred.flatten().numpy()
        # y_true_flat = y_true.flatten().numpy()
        # if y_pred.flat.shape != y_true_flat.shape:
        #     y_pref_flat = y_pred_flat[:y_true_flat.size]
        # correlation = np.corrcoef(y_pref_flat, y_true_flat)[0, 1]

        #v4
        y_pred_flat = y_pred.flatten().numpy()
        y_true_flat = y_true.flatten().numpy()
        if y_pred_flat.shape != y_true_flat.shape:
            y_pred_flat = y_pred_flat[:y_true_flat.size]
        correlation = np.corrcoef(y_pred_flat, y_true_flat)[0, 1]





        #==============================================>

        # #op6 use domain_kownledge
        # if self.domain_knowledge == "classification":
        #     is_classification = True
        # elif self.domain_knowledge == "regression":
        #     is_classification = False


        if is_classification is None:
            if self.domain_knowledge == "classification":
                is_classification = True
            elif self.domain_knowledge == "regression":
                is_classification = False

            
        #==============================================>


        #op7 analyze distribution of values in y_pred
        #assuiming certainty indicates a classification problem
        # if np.max(y_pred) > 0.9:
        #     is_classification = True


        #v2
        # if torch.max(y_pred) > 0.9:
        #     is_classification = True

        #v3
        if is_classification is None:
            if torch.max(y_pred) > 0.9:
                is_classification = True

        #==============================================>

        
        #op8 check the baalance of classes in y_true
        #assuming imbalanced classes indicate a classification problem
        # class_balance = value_counts / np.sum(value_counts)
        # if np.any(class_balance < 0.1):
        #     is_classification = True

        #v2
        if is_classification is None:
            class_balance = value_counts / np.sum(value_counts)
            if np.any(class_balance < 0.1):
                is_classification = True

        #==============================================>


        #op9 use a model selection technique
        #this optimization requires a model and a dataset so its not implemented
        #  you can implement this op outside the determine_loss_function method

        #==============================================>


        #op10 leverage user input or metadata
        # if self.user_input == "classification":
        #     is_classification = True
        # elif self.user_input == "regression":
        #     is_classification  = False


        #v2
        if is_classification is None:
            if self.user_input == "classification":
                is_classification = True
            elif self.user_input == "regression":
                is_classification = False

        #set the loss function based on the determined problem type
        if is_classification:
            self.loss_function = CrossEntropyLoss()
        else:
            self.loss_function = MSELoss()

    def compute_loss(self, y_pred, y_true):
        dataset_id = id(y_true)
        if dataset_id not in self.loss_function.cache:
            self.determine_loss_function(y_pred, y_true)
            self.loss_function_cache[dataset_id] = self.loss_function

        # self.determine_loss_function(y_pred, y_true)
        # return self.loss_function.compute_loss(y_pred, y_true)

        cached_loss_function = self.loss_function_cache[dataset_id]
        return cached_loss_function.compute_loss(y_pred, y_true)
    
        