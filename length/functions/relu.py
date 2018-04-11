import numpy as np

from length.function import Function


class Relu(Function):
    """
    The Relu Layer is a non-linear activation
    """
    name = "ReLU"

    def __init__(self):
        super().__init__()
        # TODO: add more initialization if necessary

    def internal_forward(self, inputs):
        x, = inputs
        # TODO: calculate forward pass of ReLU function
        return x,

    def internal_backward(self, inputs, gradients):
        x, = inputs
        grad_in, = gradients
        # TODO: calculate gradients of ReLU function with respect to the input
        return grad_in,


def relu(x):
    """
    This function computes the element-wise ReLU activation function (https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
    on a given input vector x.
    :param x: the input vector
    :return: a rectified version of the input vector
    """
    return Relu()(x)
