#using gradient boosted greedy algorithms to compute loss


import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Load your dataset
# X, y = load_your_data()

# For demonstration purposes, we'll use random data
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42
}

bst = xgb.train(params, dtrain, num_boost_round=100, early_stopping_rounds=10, evals=[(dtest, 'test')])

# Make predictions on the same dataset
y_pred = bst.predict(dtest)

# Determine the loss function
y_pred_labels = np.round(y_pred)
accuracy = accuracy_score(y_test, y_pred_labels)
mse = mean_squared_error(y_test, y_pred)

print("Accuracy:", accuracy)
print("Mean Squared Error:", mse)

if accuracy > 0.9:
    print("Use CrossEntropyLoss")
else:
    print("Use MSELoss")

