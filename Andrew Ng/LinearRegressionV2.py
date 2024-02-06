import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

import pandas as pd
data = pd.read_csv('CSV data/insurance.csv')

def loss_function(m, b, points):
  totalError = 0
  for i in range(len(points)):
    x = points.iloc[i].bmi
    y = points.iloc[i].charges
    totalError += (y - (m*x + b))**2
  totalError / float(len(points))

def gradient_descent(m_now, b_now, points, Lr):
  m_grad = 0
  b_grad = 0
  n = len(points)

  for i in range(n):
    x = points.iloc[i].bmi
    y = points.iloc[i].charges
    m_grad += -(2/n) * x * (y - (m_now * x + b_now))
    b_grad += -(2/n) * (y - (m_now * x + b_now))

  m = m_now - m_grad * Lr
  b = b_now - b_grad * Lr

  return m, b

m = 0
b = 0
lr = 1e6
epochs = 200

for i in range(epochs):
  if i % 50 == 0:
    print(f"epoch: {i}")
  m, b = gradient_descent(m, b, data, lr)

print(m, b)

import matplotlib.pyplot as plt

plt.scatter(data.charges, data.bmi, color="black")
plt.plot(list(range(20, 80)), [m * x + b for x in range(20,80)], color='red')
plt.show()