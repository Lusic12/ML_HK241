import numpy as np
import matplotlib.pyplot as plt

# Dataset
dataset = np.array([
    [0.245, 0], [0.247, 0], [0.285, 1], [0.299, 1], [0.327, 1], [0.347, 1],
    [0.356, 0], [0.36, 1], [0.363, 0], [0.364, 1], [0.398, 0], [0.4, 1],
    [0.409, 0], [0.421, 1], [0.432, 0], [0.473, 1], [0.509, 1], [0.529, 1],
    [0.561, 0], [0.569, 0], [0.594, 1], [0.638, 1], [0.656, 1], [0.816, 1],
    [0.853, 1], [0.938, 1], [1.036, 1], [1.045, 1],
])

# Preprocessing dataset
x = dataset[:, 0]
y = dataset[:, 1]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Prediction function
def predict(theta0, theta1, x):
    z = theta0 + theta1 * x
    return sigmoid(z)

# Cost function
def cost_function(theta0, theta1, x, y_true):
    epsilon = 1e-15  # To avoid log(0)
    m = len(x)
    y_pred = predict(theta0, theta1, x)
    cost = (-1 / m) * np.sum(
        y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)
    )
    return cost

# Gradient Descent
def gradient_descent(theta0, theta1, x, y, learning_rate, epochs):
    m = len(x)
    for _ in range(epochs):
        y_pred = predict(theta0, theta1, x)
        gradient0 = (1 / m) * np.sum(y_pred - y)
        gradient1 = (1 / m) * np.sum((y_pred - y) * x)
        theta0 -= learning_rate * gradient0
        theta1 -= learning_rate * gradient1
    final_cost = cost_function(theta0, theta1, x, y)
    return theta0, theta1, final_cost

# Initialize parameters
np.random.seed(42)
theta0 = np.random.rand()
theta1 = np.random.rand()
learning_rate = 1e-4
epochs = 10000

# Train the model
t0, t1, final_cost = gradient_descent(theta0, theta1, x, y, learning_rate, epochs)
print(f"Trained Parameters: t0 = {t0}, t1 = {t1}, Final Cost = {final_cost}")

# Prediction function for new data
def classify(theta0, theta1, x):
    probability = predict(theta0, theta1, x)
    return "present" if probability >= 0.5 else "absent"

# Test the model
test_value = 0.56
prediction = classify(t0, t1, test_value)
print(f"Prediction for {test_value}: {prediction}")

# Visualization of the dataset and decision boundary
def plot_data_and_decision_boundary(x, y, theta0, theta1):
    plt.figure(figsize=(8, 6))

    # Scatter plot of dataset
    plt.scatter(x[y == 0], y[y == 0], color='red', label='Absent', s=50)
    plt.scatter(x[y == 1], y[y == 1], color='blue', label='Present', s=50)

    # Decision boundary
    x_range = np.linspace(min(x), max(x), 500)
    y_pred = sigmoid(theta0 + theta1 * x_range)
    plt.plot(x_range, y_pred, color='green', label='Decision Boundary', linewidth=2)

    # Threshold line
    threshold_x = -theta0 / theta1
    plt.axvline(threshold_x, color='yellow', linestyle='--', label='Threshold')

    # Labels and legend
    plt.xlabel('Studying Hours')
    plt.ylabel('Predicted Probability of Pass')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_data_and_decision_boundary(x, y, t0, t1)
