import numpy as np

from length.layer import Layer
from length.constants import DTYPE
from length.initializers.xavier import Xavier


class FullyConnected(Layer):
    """
    The FullyConnected Layer is one of the base building blocks of a neural network. It computes a weighted sum
    over the input, using a weight matrix. It furthermore applies a bias term to this weighted sum to allow linear
    shifts of the computed values.
    """
    name = "FullyConnected"

    def __init__(self, num_inputs, num_outputs, weight_init=Xavier()):
        super().__init__()

        self._weights = np.zeros((num_outputs, num_inputs,), dtype=DTYPE)
        weight_init(self._weights)
        self.bias = np.zeros((num_outputs,), dtype=DTYPE)

    @property
    def weights(self):
        return self._weights.T

    @weights.setter
    def weights(self, value):
        self._weights = value.T

    def internal_forward(self, inputs):
        x, = inputs
        return np.dot(x, self._weights.T) + self.bias,

    def internal_backward(self, inputs, gradients):
        x, = inputs
        grad_in, = gradients
        grad_x = np.dot(grad_in, self._weights)
        grad_w = np.dot(grad_in.T, x)
        grad_b = np.sum(grad_in, axis=0)

        assert grad_x.shape == x.shape
        assert grad_w.shape == self._weights.shape
        assert grad_b.shape == self.bias.shape

        return grad_x, grad_w, grad_b

    def internal_update(self, parameter_deltas):
        delta_w, delta_b = parameter_deltas
        self._weights -= delta_w
        self.bias -= delta_b
