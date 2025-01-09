import numpy as np
import matplotlib.pyplot as plt
from classes import Network, Tanh, Layer, Loss

def cross_entropy_loss(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    epsilon = 1e-12
    x = np.clip(x, epsilon, 1. - epsilon)
    return -np.sum(y * np.log(x))

def cross_entropy_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x - y

def mse_loss(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mean((x - y) ** 2)

def mse_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2 * (x - y) / x.size

def plot_loss(eloss, tloss):
    plt.figure(figsize=(10, 6))
    plt.plot(eloss, label='Training Loss', color='blue')
    plt.plot(tloss, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def accuracy(net: Network, X: np.ndarray, y: np.ndarray):
  correct = 0

  for i in range(len(X)):
      output = net(X.values[i])
      predicted_label = np.argmax(output)
      true_label = y.iloc[i]

      if predicted_label == true_label:
          correct += 1
  accuracy = correct / len(X)
  print(f"Accuracy: {accuracy * 100:.2f}%")

def train_getloss(model:Network, epochs, X_train, y_train, X_test, y_test):
  model.compile(Loss(mse_loss, mse_derivative))
  e, t = model.fit(X_train.values, y_train, X_test.values, y_test, epochs, 0.01, 1)
  return e, t

def train(model:Network, epochs):
  model.compile(Loss(mse_loss, mse_derivative))
  e, t = model.fit(X_train.values, y_train_one_hot, X_test.values, y_test_one_hot, epochs, 0.01, 1)
  plot_loss(e, t)
  accuracy(model, X_test, y_test)
