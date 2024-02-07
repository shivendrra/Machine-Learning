import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

import pandas as pd
data = pd.read_csv('CSV data/Salary_Data.csv')
x_data = data['Experience']
y_data = data['Salary']

def loss_function(m, b, points):
  totalError = 0
  for i in range(len(points)):
    x = points.iloc[i].bmi
    y = points.iloc[i].charges
    totalError += (y - (m*x + b))**2
  totalError / float(len(points))

def gradient_descent(m_now, b_now, points, lr):
  m_grad = 0
  b_grad = 0
  n = len(points)

  for i in range(n):
    x = points.iloc[i].Experience
    y = points.iloc[i].Salary
    m_grad += -(2/n) * x * (y - (m_now * x + b_now))
    b_grad += -(2/n) * (y - (m_now * x + b_now))
  m = m_now - m_grad * lr
  b = b_now - b_grad * lr
  return m, b

m = 0
b = 0
lr = 1e-3
epochs = 30

for i in range(epochs):
  m, b = gradient_descent(m, b, data, lr)
  if i % 10 == 0:
    print(f"epochs: {epochs}")
    print(f"m: {m}, b: {b}")

print(f"final m: {m}, final b: {b}")

import matplotlib.pyplot as plt

plt.scatter(x_data, y_data, color="black")
plt.plot([m * x + b for x in range(len(data))], color='red')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()