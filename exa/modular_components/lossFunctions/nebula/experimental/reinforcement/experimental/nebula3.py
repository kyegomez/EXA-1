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
    
# class MSELoss(LossFunction):
#     def __init__(self):
#         self.loss_function = nn.MSELoss()

#     def compute_loss(self, y_pred, y_true):
#         return self.loss_function(y_pred, y_true)
class MSELoss(LossFunction):
    def __init__(self):
        super().__init__()
        self.loss_function = nn.MSELoss()

    def compute_loss(self, y_pred, y_true):
        y_true_one_hot = torch.zeros_like(y_pred)
        y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)
        return self.loss_function(y_pred, y_true_one_hot)



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


#reinforcement
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym 
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback
from torch.nn.utils import clip_grad_norm_


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim=features_dim)
        print(f"Observation space: {observation_space} and features_dim: {features_dim} ")


    def forward(self, observations):
        # Extract features from the observations (state representation)
        features = torch.tensor(observations).float()
        return features
    
class LossFunctionEnv(gym.Env):
    def __init__(self, y_pred, y_true):
        super().__init__()
        self.y_pred = y_pred
        self.y_true = y_true
        self.action_space = spaces.Discrete(len([CrossEntropyLoss, MSELoss]))  # Add more loss functions as needed
        self.observation_space = spaces.Box(low=0, high=float('inf'), shape=(2,), dtype=np.float32)
        self.state = self.precompute_state(y_pred, y_true)

    def precompute_state(self, y_pred, y_true):
        num_unique_values = len(torch.unique(y_true))
        pred_range = torch.max(y_pred) - torch.min(y_pred)
        state = [num_unique_values, pred_range.item()]
        return state

    def reset(self):
        # Reset the environment and return the initial state
        state = self.extract_state(self.y_pred, self.y_true)
        return state

    def step(self, action):
        # Map the action to the corresponding loss function
        loss_function = map_action_to_loss_function(action)

        # Compute the loss using the selected loss function
        loss = loss_function.compute_loss(self.y_pred, self.y_true)


        #define the reward based model on the loss
        max_loss = torch.max(self.y_pred).item() * len(self.y_pred)

        # Define the reward based on the loss
        reward = -loss.item() / max_loss

        #2nd loss 
        # reward = np.exp(-loss.item())

        # Check if the episode is done (e.g., after a certain number of steps or a certain loss threshold)
        done = False

        # Return the next state, reward, and done flag
        next_state = self.extract_state(self.y_pred, self.y_true)
        return next_state, reward, done, {}

    def extract_state(self, y_pred, y_true):
        num_unique_values = len(torch.unique(y_true))
        pred_range = torch.max(y_pred) - torch.min(y_pred)
        state = [num_unique_values, pred_range.item()]
        return state
    
#Cache action-to-loss-function mapping
action_to_loss_function_cache = {}

def map_action_to_loss_function(action):
    # action = int(action[0])
    if isinstance(action, np.ndarray):
        action = action.item()
    if action not in action_to_loss_function_cache:
        if action == 0:
            action_to_loss_function_cache[action] = CrossEntropyLoss()
        elif action == 1:
            action_to_loss_function_cache[action] = MSELoss()
    
    return action_to_loss_function_cache[action]

    #add more loss functions as needed

# Create a DummyVecEnv wrapper for the LossFunctionEnv
def make_env(y_pred, y_true):
    def _init():
        return LossFunctionEnv(y_pred, y_true)
    return _init


y_pred = torch.randn(100, 10)
y_true = torch.randint(0, 10, (100,))

env = DummyVecEnv([make_env(y_pred, y_true)])

# Create a custom policy network that uses the CustomFeaturesExtractor
policy_kwargs = dict(
    features_extractor_class=CustomFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=env.observation_space.shape[0]),  # Define the observation space based on the state representation
)


class CustomCallback(BaseCallback):
    def __init__(self, patience, min_delta, initial_lr, decay_rate, decay_steps, max_norm, verbose=0):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.max_norm = max_norm
        self.counter = 0
        self.previous_performance = -np.inf
        self.stop_training = False

    def _on_step(self) -> bool:
        # Early stopping
        performance = self.model.ep_info_buffer.mean()
        if performance - self.previous_performance < self.min_delta:
            self.counter += 1
        else:
            self.counter = 0
        if self.counter >= self.patience:
            self.stop_training = True
        self.previous_performance = performance

        # Learning rate scheduler
        step = self.num_timesteps
        lr = self.initial_lr * (self.decay_rate ** (step / self.decay_steps))
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient clipping
        clip_grad_norm_(self.model.policy.parameters(), self.max_norm)

        return not self.stop_training
    

callback = CustomCallback(patience=10, min_delta=0.001, initial_lr=0.01, decay_rate=0.9, decay_steps=1000, max_norm=1.0)
agent.learn(total_timesteps=10000, callback=callback)




# Train the final agent with the best hyperparameters
agent = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
agent.learn(total_timesteps=10000)

# Use the trained agent in the NebulaOptimized class
class NebulaOptimized(Nebula):
    def __init__(self, domain_knowledge=None, user_input=None):
        super().__init__(domain_knowledge, user_input)
        self.rl_agent = agent

    def determine_loss_function(self, y_pred, y_true):
        # Extract state representation from the data and model
        # state = ...  # Extract state representation from y_pred and y_true
        state = self.extract_state(y_pred, y_true)

        # Use the RL agent to select the optimal loss function
        action, _ = self.rl_agent.predict(state, deterministic=True)

        # Map the action to the corresponding loss function
        self.loss_function = map_action_to_loss_function(action)
    
    def extract_state(self, y_pred, y_true):
        num_unique_values = len(torch.unique(y_true))
        pred_range = torch.max(y_pred) - torch.min(y_pred)
        state = [num_unique_values, pred_range.item()]
        return state
    

nebula_optimized = NebulaOptimized()

#test the trained agent with new y_pred and y_true tensors
y_pred_test = torch.randn(100, 10)
y_true_test = torch.randint(0, 10, (100,))

nebula_optimized.determine_loss_function(y_pred_test, y_true_test)

print(f"Selected loss function {nebula_optimized.loss_function}")


"""

time/fps: Frames per second (FPS) measures the number of training steps the agent takes per second. Higher FPS indicates faster training. You should focus on increasing this metric to speed up the training process.

time/iterations: The number of training iterations completed by the agent. This metric increases as the agent trains. You should focus on increasing this metric to ensure the agent has enough training experience.

time/time_elapsed: The total time elapsed (in seconds) since the start of the training process. You should focus on lowering this metric to reduce the overall training time.

time/total_timesteps: The total number of timesteps the agent has experienced during training. You should focus on increasing this metric to ensure the agent has enough training experience.

train/entropy_loss: The entropy loss encourages exploration by penalizing deterministic policies. Lower entropy loss indicates less exploration. You should focus on balancing this metric to ensure the agent explores the environment sufficiently without getting stuck in suboptimal solutions.

train/explained_variance: Explained variance measures how well the agent's value function approximates the true value function. A value of 1 indicates a perfect approximation, while a value of 0 indicates no correlation. You should focus on increasing this metric to improve the agent's value function approximation.

train/learning_rate: The learning rate determines the step size for updating the agent's parameters during training. You should focus on finding the optimal learning rate that balances convergence speed and stability.

train/n_updates: The number of parameter updates performed by the agent during training. This metric increases as the agent trains. You should focus on increasing this metric to ensure the agent has enough training experience.

train/policy_loss: The policy loss measures the difference between the agent's predicted actions and the optimal actions. You should focus on lowering this metric to improve the agent's policy.

train/value_loss: The value loss measures the difference between the agent's predicted state values and the true state values. You should focus on lowering this metric to improve the agent's value function approximation.

In summary, you should focus on increasing the metrics related to training experience (iterations, total_timesteps, n_updates) and performance (explained_variance), while lowering the metrics related to training time (time_elapsed) and losses (policy_loss, value_loss). Balancing exploration (entropy_loss) and finding the optimal learning rate are also important for achieving good performance.


"""