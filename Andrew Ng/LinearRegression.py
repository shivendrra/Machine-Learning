import os
os.chdir('d:\Machine learning\Machine Learning Git-repo\Andrew Ng')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('CSV data/insurance.csv')
X = dataset[['bmi']].values
y = dataset['charges'].values 

X_b = np.c_[np.ones((len(X), 1)), X]
theta = np.zeros((X_b.shape[1], 1))
learning_rate = 16e-6
max_iter = 5000

def mean_squared_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, theta, learning_rate):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    theta = theta - (1 / m) * learning_rate * X.T.dot(errors)
    cost = mean_squared_cost(X, y, theta)
    return theta, cost

def optimizer(X, y, theta, learning_rate, max_iterations, tolerance=1e-6):
    cost_history = []

    for i in range(max_iterations):
        theta, cost = gradient_descent(X, y, theta, learning_rate)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f'Iteration {i}, Cost: {cost}')

        if len(cost_history) > 1 and abs(cost_history[-2] - cost_history[-1]) < tolerance:
            print(f'Converged at iteration {i}, Cost: {cost}')
            break

    return theta, cost_history

# theta, cost_history = gradient_descent(X_b, y, theta, learning_rate, iterations)

theta, cost_history = optimizer(X_b, y, theta, learning_rate, max_iterations=max_iter)

plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression')
plt.show()

print('weights:', theta)
print('Final Cost:', cost_history[-1])