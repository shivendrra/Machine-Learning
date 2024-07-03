import numpy as np

class MLP:
  def __init__(self, input_size, hidden_size, output_size):

    self.W1 = np.random.randn(hidden_size, input_size) * 0.01
    self.b1 = np.zeros((hidden_size, 1))
    self.W2 = np.random.randn(output_size, hidden_size) * 0.01
    self.b2 = np.zeros((output_size, 1))
  
  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
  
  def sigmoid_derivative(self, z):
    return z * (1 - z)
  
  def forward(self, X):
    self.Z1 = np.dot(self.W1, X) + self.b1
    self.A1 = self.sigmoid(self.Z1)
    self.Z2 = np.dot(self.W2, self.A1) + self.b2
    self.A2 = self.sigmoid(self.Z2)
    return self.A2
  
  def backward(self, X, Y, learning_rate):
    m = X.shape[1]

    # Calculate the gradients
    dZ2 = self.A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
      
    dA1 = np.dot(self.W2.T, dZ2)
    dZ1 = dA1 * self.sigmoid_derivative(self.A1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
      
    # Update the weights
    self.W1 -= learning_rate * dW1
    self.b1 -= learning_rate * db1
    self.W2 -= learning_rate * dW2
    self.b2 -= learning_rate * db2
  
  def train(self, X, Y, iters, learning_rate):
    for i in range(iters):
      output = self.forward(X)
      self.backward(X, Y, learning_rate)
      if i % 100 == 0:
        loss = self.calculate_loss(Y, output)
        print(f"Iteration {i}, loss: {loss}")  
  
  def calculate_loss(self, Y, output):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(output) + (1 - Y) * np.log(1 - output)) / m
    return loss

np.random.seed(42)
X = np.random.randn(2, 100)
Y = (np.sum(X, axis=0) > 0).reshape(1, 100)

mlp = MLP(input_size=2, hidden_size=3, output_size=1)
mlp.train(X, Y, iters=1000, learning_rate=0.1)