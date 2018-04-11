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
        # HINT: some ideas for the implementation can be found here:
        # https://cs231n.github.io/neural-networks-1/#layers

        # TODO: initialize weights with correct shape, using the weight initializer 'weight_init'
        self._weights = None
        # TODO: initialize bias with correct shape and correct initial values
        self.bias = None

    @property
    def weights(self):
        # TODO: (if necessary) add code to transform between internal and external representation
        return self._weights

    @weights.setter
    def weights(self, value):
        # TODO: (if necessary) add code to transform between internal and external representation
        self._weights = value

    def internal_forward(self, inputs):
        x, = inputs
        # TODO: calculate result for this layer
        # HINT: the expected shape of the result is (usually) not equal to the shape of x
        result = x
        return result,

    def internal_backward(self, inputs, gradients):
        x, = inputs
        grad_in, = gradients

        # TODO: calculate gradients with respect to inputs for this layer
        grad_x = None
        grad_w = None
        grad_b = None

        assert grad_x.shape == x.shape
        # the gradients of the weights should have the same shape as the internal weights array for the test case
        assert grad_w.shape == self._weights.shape
        assert grad_b.shape == self.bias.shape

        return grad_x, grad_w, grad_b

    def internal_update(self, parameter_deltas):
        delta_w, delta_b = parameter_deltas
        # TODO: apply updates to weights and bias according to deltas from optimizer
        self._weights = self._weights
        self.bias = self.bias
