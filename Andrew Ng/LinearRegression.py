import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Convert to Pandas DataFrame
data = pd.DataFrame({'X': X.flatten(), 'y': y.flatten()})

# Linear Regression using Gradient Descent
def mean_squared_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta = theta - (1 / m) * learning_rate * X.T.dot(errors)
        cost_history[i] = mean_squared_cost(X, y, theta)

    return theta, cost_history

# adding bias to x
X_b = np.c_[np.ones((100, 1)), X]

# parameters initialized
theta = np.random.randn(2, 1)

# hyperparameters
learning_rate = 0.01
iterations = 1000

# gradient descent
theta, cost_history = gradient_descent(X_b, y, theta, learning_rate, iterations)

# Plot the data and the linear regression line
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.show()

# Print the final parameters and cost history
print('Theta:', theta)
print('Final Cost:', cost_history[-1])
