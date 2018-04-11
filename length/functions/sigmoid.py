import numpy as np

from length.function import Function


class Sigmoid(Function):
    name = "Sigmoid"

    def __init__(self):
        super().__init__()
        # TODO: add more initialization if necessary

    def internal_forward(self, inputs):
        x, = inputs
        # TODO: calculate and return result of sigmoid function
        output = x
        return output,

    def internal_backward(self, inputs, gradients):
        x, = inputs
        grad_in, = gradients
        # TODO: calculate the gradients of this function with respect to its inputs
        grad_x = grad_in
        assert grad_x.shape == x.shape
        return grad_x,


def sigmoid(x):
    """
    This function computes the element-wise sigmoid activation function (https://en.wikipedia.org/wiki/Sigmoid_function)
    on a given input x.
    :param x: the vector, where the activation function shall be applied to
    :return: a vector with the sigmoid activation applied
    """
    return Sigmoid()(x)
