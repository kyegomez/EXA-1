import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from v2 import calabi_yau_loss

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with cross-entropy loss
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Evaluate the model using cross-entropy loss
y_pred = model.predict(X_test)
cross_entropy_loss = log_loss(y_test, y_pred)

# Generate random perturbations for the stability component of the Calabi-Yau inspired loss function
perturbations = [np.random.normal(0, 0.1, X_test.shape) for _ in range(5)]

# Evaluate the model using the Calabi-Yau inspired loss function
calabi_yau_loss_value = calabi_yau_loss(model, X_test, y_test, perturbations)

print(f'Cross-entropy loss: {cross_entropy_loss}')
print(f'Calabi-Yau inspired loss: {calabi_yau_loss_value}')