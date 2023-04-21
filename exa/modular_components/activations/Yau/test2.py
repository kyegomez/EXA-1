# import numpy as np
# import tensorflow as tf
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import log_loss
# # from v2 import calabi_yau_loss
# # from polymorphic import calabi_yau_loss
# from polymorphicv2 import calabi_yau_loss

# # Create a synthetic dataset
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a simple neural network model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(16, activation='relu', input_shape=(20,)),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model with cross-entropy loss
# model.compile(optimizer='adam', loss='binary_crossentropy')

# # Train the model
# model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# # Evaluate the model using cross-entropy loss
# y_pred = model.predict(X_test)
# cross_entropy_loss = log_loss(y_test, y_pred)

# # Generate random perturbations for the stability component of the Calabi-Yau inspired loss function
# perturbations = [np.random.normal(0, 0.1, X_test.shape) for _ in range(5)]

# # Update y_pred and y_true to be 2D arrays for topological_invariance function
# y_pred_2d = np.hstack((X_test, y_pred))
# y_true_2d = np.hstack((X_test, y_test.reshape(-1, 1)))

# # Evaluate the model using the Calabi-Yau inspired loss function
# calabi_yau_loss_value = calabi_yau_loss(model, X_test, y_test, perturbations)

# print(f'Cross-entropy loss: {cross_entropy_loss}')
# print(f'Calabi-Yau inspired loss: {calabi_yau_loss_value}')


import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from polymorphicv2 import calabi_yau_loss
# from p3 import calabi_yau_loss

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32)




# Create a simple neural network model
model = torch.nn.Sequential(
    torch.nn.Linear(20, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1),
    torch.nn.Sigmoid()
)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.BCELoss()

# Train the model
for epoch in range(10):
    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate the model using cross-entropy loss
y_pred = model(X_test)
cross_entropy_loss = log_loss(y_test, y_pred.detach().numpy())

# Generate random perturbations for the stability component of the Calabi-Yau inspired loss function
perturbations = [torch.tensor(np.random.normal(0, 0.1, X_test.shape), dtype=torch.float32) for _ in range(5)]

# Update y_pred and y_true to be 2D arrays for topological_invariance function
y_pred_2d = torch.cat((X_test, y_pred), dim=1)
y_true_2d = torch.cat((X_test, y_test.view(-1, 1)), dim=1)

# Evaluate the model using the Calabi-Yau inspired loss function
calabi_yau_loss_value = calabi_yau_loss(model, X_test, y_test, perturbations)

print(f'Cross-entropy loss: {cross_entropy_loss}')
print(f'Calabi-Yau inspired loss: {calabi_yau_loss_value}')
