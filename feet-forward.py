import numpy as np
from typing import Callable
from sklearn.datasets import load_digits
import numpy.typing as npt

import functools
from math import exp


class ActivationFunction:
    """
    Activation function class.

    Attributes:
        foo (func): vectorized activation function.
        dfoo (func): vectorized derivative of activation function.
    """

    def __init__(self, foo: Callable, dfoo: Callable):
        self.foo = np.vectorize(foo)
        self.dfoo = np.vectorize(dfoo)

    def __call__(self, *args, **kwargs):
        return self.foo(*args, **kwargs)


class Layer:
    """
    Single layer.

    Attributes:
        weights (np.ndarray): matrix of weights.
    """

    def __init__(self, dim_in: int, dim_out: int, activation_foo: ActivationFunction, weights=None, biases=None):
        if weights is None:
            self.weights = np.random.normal(0, 2 / (dim_in + dim_out), (dim_out, dim_in))
            self.biases = np.random.normal(0, 2 / (dim_in + dim_out), (dim_out, 1))
        else:
            self.weights = weights
            self.biases = biases
        self.activation_foo = activation_foo

    def calculate(self, data_in: np.ndarray):
        multiple_biases = np.hstack([self.biases] * data_in.shape[1])
        return self.activation_foo(self.weights @ data_in + multiple_biases)

    def sub(self, gradient):
        self.weights -= gradient.weights
        self.biases -= gradient.biases

    def mulmore(self, other):
        self.weights *= other
        self.biases *= other
        return self


class LossFunction:
    def __init__(self, function: Callable = None, gradient=None):
        if function:
            self.function = function
            self.gradient = gradient
        else:
            self.function = lambda result, target: np.linalg.norm(target - result)**2/2
            self.gradient = lambda result, target: target - result

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class Net:
    def __init__(self, lay_dims: list[int], activation_foo: ActivationFunction,
                 last_activation_foo: ActivationFunction = None, loss_function=LossFunction()):
        self.layers = [Layer(dim_in, dim_out, activation_foo) for dim_in, dim_out in zip(lay_dims[:-1], lay_dims[1:])]
        if last_activation_foo is not None:
            self.layers[-1].activation_foo = last_activation_foo
        self.loss_function = loss_function

    def calculate(self, data_in):
        papiesz = [data_in] + list(self.layers)
        return functools.reduce(lambda v, layer: layer.calculate(v), papiesz)

    def backprop_calculate(self, data_in):
        results = []
        last = data_in
        for layer in self.layers:
            result = layer.calculate(last)
            results.append(result)
            last = result
        return results

    def backpropagation(self, batch: np.ndarray):
        partial_results = self.backprop_calculate(batch)

    def training_step(self, batch: np.ndarray, learning_rate: float) -> None:
        gradients = self.backpropagation(batch)
        map(lambda layer_grad: layer_grad[0].sub(layer_grad[1].mulmore(learning_rate)), zip(self.layers, gradients))

    def train(self, training_data: np.ndarray, training_labels: np.ndarray, epochs: int, batch_size: int,
              learning_rate_sequence: Callable):
        for i in range(epochs):
            batch = np.random.choice(zip(training_data, training_labels), batch_size)
            self.training_step(batch, learning_rate_sequence(i))


if __name__ == '__main__':
    foo = ActivationFunction(lambda x: max(x, 0), lambda x: int(x > 0))
    last_foo = ActivationFunction(lambda x: 1 / (1 + exp(-x)), lambda x: 1 / (1 + exp(-x)) * (1 - 1 / (1 + exp(-x))))
    net = Net((64, 32, 32, 10), foo, last_foo)
    random_input = np.random.random((64, 1))
    print(net.calculate(random_input))
