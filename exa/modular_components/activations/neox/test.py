import torch
import torch.nn as nn
from neox2 import SimplifiedOptimizedFractionalActivation
# import torch
# import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
# from neox3 import CaputoFractionalActivation
# from neox4 import CaputoFractionalActivation
# from neox5 import CaputoFractionalActivation
# from neox6 import CaputoFractionalActivation
from neo7 import CaputoFractionalActivation
from torch.utils.data import DataLoader

#test neox5

class NovelActivationModel(nn.Module):
    def __init__(self):
        super(NovelActivationModel, self).__init__()
        self.fc1 = nn.Linear(20, 128)
        self.act1 = CaputoFractionalActivation(nn.ReLU(), 0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

class ReLUModel(nn.Module):
    def __init__(self):
        super(ReLUModel, self).__init__()
        self.fc1 = nn.Linear(20, 128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

novel_activation_model = NovelActivationModel()
relu_model = ReLUModel()

# 1. Generate synthetic dataset
num_samples = 1000
input_features = 20
output_classes = 2

# 1a. Generate random input features
X = torch.randn(num_samples, input_features)

# 1b. Generate corresponding output labels based on a predefined rule or function
y = (X.sum(dim=1) > 0).long()

# 2. Split the dataset into training and validation sets
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 3. Create the neural network models
novel_activation_model = NovelActivationModel()
relu_model = ReLUModel()

# 4. Define the loss function, optimizer, and other hyperparameters
criterion = nn.CrossEntropyLoss()
novel_optimizer = optim.Adam(novel_activation_model.parameters(), lr=0.001)
relu_optimizer = optim.Adam(relu_model.parameters(), lr=0.001)
epochs = 50
batch_size = 64

# # 5. Train the models
# def train_model(model, optimizer, train_dataset, val_dataset, criterion, epochs, batch_size):
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)

#     for epoch in range(epochs):
#         model.train()
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#         model.eval()
#         val_loss = 0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         accuracy = correct / total
#         print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.4f}')


from torch.utils.data import DataLoader

def train_model(model, optimizer, train_dataset, val_dataset, criterion, epochs, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            val_loss /= len(val_loader)
            accuracy = correct / total
            print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')


train_model(novel_activation_model, novel_optimizer, train_dataset, val_dataset, criterion, epochs, batch_size)
train_model(relu_model, relu_optimizer, train_dataset, val_dataset, criterion, epochs, batch_size)

# 6. Evaluate the models on the validation set and compare their performance
# The evaluation is done during the training process, and the performance is printed at each epoch.