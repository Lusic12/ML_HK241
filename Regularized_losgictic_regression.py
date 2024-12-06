import numpy as np
import matplotlib.pyplot as plt

# Preprocessing the dataset
dataset = np.array([
    [0.245, 0], [0.247, 0], [0.285, 1], [0.299, 1], [0.327, 1], [0.347, 1],
    [0.356, 0], [0.36, 1], [0.363, 0], [0.364, 1], [0.398, 0], [0.4, 1],
    [0.409, 0], [0.421, 1], [0.432, 0], [0.473, 1], [0.509, 1], [0.529, 1],
    [0.561, 0], [0.569, 0], [0.594, 1], [0.638, 1], [0.656, 1], [0.816, 1],
    [0.853, 1], [0.938, 1], [1.036, 1], [1.045, 1],
])

x1 = dataset[:, 0]
x2 = x1**2
y = dataset[:, 1]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Prediction function
def predict(theta0, theta1, theta2, x1, x2):
    z = theta0 + theta1 * x1 + theta2 * x2
    return sigmoid(z)

# Cost function
def cost_function(theta0, theta1, theta2, x1, x2, y_true, lambda_reg):
    epsilon = 1e-15  # Avoid division by zero
    m = len(x1)
    y_pred = predict(theta0, theta1, theta2, x1, x2)
    reg_term = (lambda_reg / (2 * m)) * (theta1**2 + theta2**2)
    cost = (-1 / m) * np.sum(
        y_true * np.log(y_pred + epsilon) +
        (1 - y_true) * np.log(1 - y_pred + epsilon)
    ) + reg_term
    return cost

# Gradient descent function
def gradient_descent(theta0, theta1, theta2, x1, x2, y, learning_rate, epochs, lambda_reg):
    m = len(x1)
    for _ in range(epochs):
        y_pred = predict(theta0, theta1, theta2, x1, x2)
        gradient0 = (1 / m) * np.sum(y_pred - y)
        gradient1 = (1 / m) * np.sum((y_pred - y) * x1) + (lambda_reg / m) * theta1
        gradient2 = (1 / m) * np.sum((y_pred - y) * x2) + (lambda_reg / m) * theta2
        theta0 -= learning_rate * gradient0
        theta1 -= learning_rate * gradient1
        theta2 -= learning_rate * gradient2
    final_cost = cost_function(theta0, theta1, theta2, x1, x2, y, lambda_reg)
    return theta0, theta1, theta2, final_cost

# Initial parameters
np.random.seed(42)
theta0 = np.random.rand()
theta1 = np.random.rand()
theta2 = np.random.rand()
learning_rate = 1e-4
epochs = 10000
lambda_reg = 0.5

# Training the model
t0, t1, t2, final_cost = gradient_descent(theta0, theta1, theta2, x1, x2, y, learning_rate, epochs, lambda_reg)
print(f"Trained parameters: t0 = {t0}, t1 = {t1}, t2 = {t2}, Final cost = {final_cost}")

# Prediction function for new data
def classify(t0, t1, t2, x):
    prob = predict(t0, t1, t2, x, x**2)
    return "present" if prob >= 0.5 else "absent"

# Test a prediction
test_value = 0.56
result = classify(t0, t1, t2, test_value)
print(f"Prediction for {test_value}: {result}")

# Visualization of the dataset and classification curve
def plot_dataset_and_decision_boundary(x1, y, t0, t1, t2):
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of dataset
    plt.scatter(x1[y == 0], y[y == 0], color='red', label='Absent', s=50)
    plt.scatter(x1[y == 1], y[y == 1], color='blue', label='Present', s=50)
    
    # Decision boundary
    x_range = np.linspace(min(x1), max(x1), 500)
    y_pred = sigmoid(t0 + t1 * x_range + t2 * x_range**2)
    plt.plot(x_range, y_pred, color='green', linewidth=2, label='Decision Boundary')
    
    plt.axhline(0.5, color='yellow', linestyle='--', label='Threshold')
    plt.xlabel('Grain Size')
    plt.ylabel('Predicted Probability')
    plt.title('Grain Size vs Predicted Presence of Spiders')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_dataset_and_decision_boundary(x1, y, t0, t1, t2)
