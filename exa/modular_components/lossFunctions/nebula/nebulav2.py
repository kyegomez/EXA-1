import torch 
import torch.nn as nn
# import torch.jit
import numpy as np

class LossFunction:
    def compute_Loss(self, y_pred, y_true):
        raise NotImplemented("compute_loss method must be implemented!")
    
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
    
def one_hot_encoding(y_true, num_classes):
    y_true_one_hot = torch.zeros(y_true.size(0), num_classes)
    y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)
    return y_true_one_hot


class SmoothL1Loss(LossFunction):
    def __init__(self):
        self.loss_function = nn.SmoothL1Loss()

    def compute_Loss(self, y_pred, y_true):
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
        return self.loss_function(F.log_softmax(y_pred, dim=1))
    
class NLLLoss(LossFunction):
    def __init__(self):
        self.loss_function = nn.NLLLoss()

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)
    
class HashableTensorWrapper:
    def __init__(self, tensor):
        self.tensor = tensor
        self.tensor_shape = tensor.shape
        self.tensor_dtype = tensor.dtype

    def __hash__(self):
        return hash((self.tensor_shape, self.tensor_dtype))

    def __eq__(self, other):
        return isinstance(other, HashableTensorWrapper) and self.tensor_shape == other.tensor_shape and self.tensor_dtype == other.tensor_dtype



#detector helper function
# def is_multi_label_classification(y_true: torch.Tensor):
#     return y_true.shape[1] > 1 and y_true.dtype == torch.float

#v2
def is_multi_label_classification(y_true: torch.Tensor) -> bool:
    return len(y_true.shape) > 1 and y_true.shape[1] > 1 and y_true.dtype == torch.float


def contains_non_negative_integers(y_true):
    return torch.all(y_true >= 0) and torch.all(y_true == y_true.to(torch.int64))

def are_probability_distributions(y_pred, y_true):
    return torch.all(y_pred >= 0) and torch.all(y_pred <= 1) and torch.all(y_true >= 0) and torch.all(y_true <= 1) 

def are_log_probabilities(y_pred):
    return torch.all(y_pred <= 0)


#generate unique key for a tensor
#v1
# def generate_tensor_key(tensor):
#     return (tuple(tensor.shape), str(tensor.dtype))

#v2 
# def generate_tensor_key(tensor):
    # shape_tuple = ()
    # for dim in tensor.shape:
    #     shape_tuple += (dim.item(),)
    # return (shape_tuple, str(tensor.dtype))


#v3
# def generate_tensor_key(tensor):
#     shape_tuple = ()
#     for dim in tensor.shape:
#         shape_tuple += (dim,)
#     return (shape_tuple, str(tensor.dtype))


#v4 
def generate_tensor_key(tensor):
    shape_tuple = ()
    for dim in tensor.shape:
        shape_tuple += (dim,)
    return (shape_tuple, str(tensor.dtype))


class CrossEntropyLoss(LossFunction):
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss()

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)
    

class Nebula(LossFunction):
    def __init__(self, domain_knowledge=None, user_input=None):
        self.loss_function = None
        self.domain_knowledge = domain_knowledge
        self.user_input = user_input 
        self.loss_function_cache = {}
        self.unique_values_cache = {}
        self.class_balance_cache = {}

    def determine_loss_function(self, y_pred, y_true):
        is_classification = None
        dataset_id = id(y_true)

        # Cache unique values
        if dataset_id not in self.unique_values_cache:
            self.unique_values_cache[dataset_id] = torch.unique(y_true)
        unique_values = self.unique_values_cache[dataset_id]

        # Cache class balance
        if dataset_id not in self.class_balance_cache:
            value_counts = torch.bincount(y_true.flatten().to(dtype=torch.int64))
            self.class_balance_cache[dataset_id] = value_counts / torch.sum(value_counts)
        class_balance = self.class_balance_cache[dataset_id]

        # Optimization 2: Use PyTorch functions instead of NumPy
        value_counts = torch.bincount(y_true.flatten().to(dtype=torch.int64))

        # The remaining code remains unchanged as it already incorporates the suggested optimizations
        if is_classification is None:
            if len(unique_values) <= 10 and torch.all(torch.eq(unique_values % 1, 0)):
                is_classification = True

        if is_classification is None:
            if torch.all(value_counts > 0):
                is_classification = True

        if y_pred.ndim > 2:
            pass

        if is_classification is None:
            sparsity = torch.count_nonzero(y_true) / y_true.numel()
            if sparsity < 0.5:
                self.loss_function = torch.nn.BCEWithLogitsLoss()
                self.compute_loss = self.loss_function
                return

        y_pred_flat = y_pred.flatten()
        y_true_flat = y_true.flatten()
        if y_pred_flat.shape != y_true_flat.shape:
            y_pred_flat = y_pred_flat[:y_true_flat.numel()]
        correlation = torch.tensor(np.corrcoef(y_pred_flat.cpu().numpy(), y_true_flat.cpu().numpy())[0, 1])

        if is_classification is None:
            if self.domain_knowledge == "classification":
                is_classification = True
            elif self.domain_knowledge == "regression":
                is_classification = False

        if is_classification is None:
            if torch.max(y_pred) > 0.9:
                is_classification = True

        if is_classification is None:
            if torch.any(class_balance < 0.1):
                is_classification = True

        if is_classification is None:
            if self.user_input == "classification":
                is_classification = True
            elif self.user_input == "regression":
                is_classification = False


        #Multi-LabelClassification
        if is_multi_label_classification(y_true):
            self.loss_function = MultiLabelSoftMarginLoss()

        #poissonNLLLoss
        if contains_non_negative_integers(y_true):
            self.loss_function = PoissonNLLoss()

        #KLDIvLoss
        if are_probability_distributions(y_pred, y_true):
            self.loss_function = KLDivLoss()


        #NLLLoss
        if is_classification and are_log_probabilities(y_pred):
            self.loss_function = NLLLoss()




        # SmotthL1Loss
        if is_classification is None:
            #check range of values in y_true
            if torch.min(y_true) >= 0 and torch.max(y_true) <= 1:
                self.loss_function = SmoothL1Loss()
        


        # Set the loss function based on the determined problem type
        if is_classification:
            self.loss_function = CrossEntropyLoss()
        else:
            self.loss_function = MSELoss()


        

        
    # @torch.jit.script #optimization jit 
    def compute_loss(self, y_pred, y_true):
        #v2
        # tensor_key = HashableTensorWrapper(y_true)
        # if tensor_key not in self.loss_function_cache:
        #     self.determine_loss_function(y_pred, y_true)
        # return self.loss_function_cache[tensor_key](y_pred, y_true)
    


        # V1
        dataset_id = id(y_true)
        if dataset_id not in self.loss_function_cache:
            self.determine_loss_function(y_pred, y_true)
            self.loss_function_cache[dataset_id] = self.loss_function
        
        cached_loss_function = self.loss_function_cache[dataset_id]
        return cached_loss_function.compute_loss(y_pred, y_true)

        #v3
        # tensor_key = generate_tensor_key(y_true)
        # if tensor_key not in self.loss_function_cache:
        #     self.determine_loss_function(y_pred, y_true)
        # return self.loss_function_cache[tensor_key](y_pred, y_true)
    
    

#move tensors nd model to gpu if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# y_pred, y_true = y_pred.to(device), y_true.to(device)


# # #example usage with the pytorch autograd profiler
# with torch.autograd.profiler.profile() as prof:
#     loss = Nebula.compute_loss(y_pred, y_true)
# print(prof.key_average().table())