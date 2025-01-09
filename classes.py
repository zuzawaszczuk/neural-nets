from abc import abstractmethod, ABC
from typing import List
import numpy as np
import matplotlib.pyplot as plt


class Layer(ABC):
    """Basic building block of the Neural Network"""

    def __init__(self) -> None:
        self._learning_rate = 0.01

    @abstractmethod
    def forward(self, x:np.ndarray)->np.ndarray:
        """Forward propagation of x through layer"""
        pass

    @abstractmethod
    def backward(self, output_error_derivative) ->np.ndarray:
        """Backward propagation of output_error_derivative through layer"""
        pass

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        assert learning_rate < 1, f"Given learning_rate={learning_rate} is larger than 1"
        assert learning_rate > 0, f"Given learning_rate={learning_rate} is smaller than 0"
        self._learning_rate = learning_rate


class FullyConnected(Layer):
    def __init__(self, input_size:int, output_size:int, init_zero:bool=False, seed:int=1) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        if init_zero:
          self.weights = np.zeros((input_size, output_size))
        else:
          np.random.seed(seed)
          self.weights = np.random.uniform(-1/np.sqrt(input_size), 1/np.sqrt(input_size), (input_size, output_size))
        self.bias = np.ones(output_size)
        self.grad_weights = np.zeros((input_size, output_size))
        self.grad_bias = np.zeros(output_size)
        self.saved_input = np.zeros(input_size)

    def forward(self, x:np.ndarray)->np.ndarray:
        self.saved_input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, output_error_derivative)->np.ndarray:
        self.grad_bias = output_error_derivative
        self.grad_weights += np.outer(self.saved_input, output_error_derivative)

        return np.dot(output_error_derivative, self.weights.T)

    def zero_grad(self):
        self.grad_weights = np.zeros((self.input_size, self.output_size))
        self.grad_bias = np.zeros(self.output_size)


class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:np.ndarray)->np.ndarray:
        self.saved_output = np.tanh(x)
        return self.saved_output

    def backward(self, output_error_derivative)->np.ndarray:
        tanh_derivative = 1 - self.saved_output**2
        return tanh_derivative * output_error_derivative


class Loss:
    def __init__(self, loss_function:callable, loss_function_derivative:callable)->None:
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def loss(self, x:np.ndarray, y:np.ndarray)->np.ndarray:
        """Loss function for a particular x"""
        return self.loss_function(x, y)

    def loss_derivative(self, x:np.ndarray, y:np.ndarray)->np.ndarray:
        """Loss function derivative for a particular x and y"""
        return self.loss_function_derivative(x, y)


class Network:
    def __init__(self, layers:List[Layer], learning_rate:float)->None:
        self.layers = layers
        self.learning_rate = learning_rate

    def compile(self, loss:Loss)->None:
        """Define the loss function and loss function derivative"""
        self.loss = loss

    def __call__(self, x:np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers"""
        for layer in self.layers:
          x = layer.forward(x)
        return x

    def fit(self,
            x_train:np.ndarray,
            y_train:np.ndarray,
            x_test:np.ndarray,
            y_test:np.ndarray,
            epochs:int,
            learning_rate:float,
            verbose:int=0)->None:
        """Fit the network to the training data"""
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(epochs):
            train_loss = 0.0
            correct_train = 0

            for i in range(len(x_train)):
                for layer in self.layers:
                    if isinstance(layer, FullyConnected):
                        layer.zero_grad()

                output = self(x_train[i])

                loss = self.loss.loss(output, y_train[i])

                loss_derivative = self.loss.loss_derivative(output, y_train[i])
                for layer in reversed(self.layers):
                    loss_derivative = layer.backward(loss_derivative)

                for layer in self.layers:
                    if isinstance(layer, FullyConnected):
                        layer.weights -= self.learning_rate * layer.grad_weights
                        layer.bias -= self.learning_rate * layer.grad_bias
                train_loss += loss

                if output.argmax() == y_train[i].argmax(): 
                    correct_train += 1

            if verbose:
                print(f'Epoch: {epoch + 1} Training Loss: {train_loss / len(x_train)}')

            train_losses.append(train_loss / len(x_train))
            train_accuracies.append(correct_train / len(x_train))

            test_loss = 0.0
            correct_test = 0
            for i in range(len(x_test)):
                output = self(x_test[i])
                test_loss += self.loss.loss(output, y_test[i])
                if output.argmax() == y_test[i].argmax(): 
                    correct_test += 1
            test_losses.append(test_loss / len(x_test))
            test_accuracies.append(correct_test / len(x_test))

        return train_losses, test_losses