import numpy as np

from length.function import Function
from length.functions.softmax import log_softmax


class SoftmaxCrossEntropy(Function):
    name = "SoftmaxCrossEntropy"

    def __init__(self):
        super().__init__()
        self.y = None

    def internal_forward(self, inputs):
        x, t = inputs
        self.y = log_softmax(x)
        loss = -self.y[range(len(t)), t].sum(keepdims=True) / t.size
        return loss[0],

    def internal_backward(self, inputs, gradients):
        x, t = inputs
        grad_in, = gradients
        grad_x = np.exp(self.y.copy())
        grad_x[range(len(t)), t] -= 1
        grad_x *= grad_in[0] / t.size
        return grad_x, None


def softmax_cross_entropy(x, t):
    """
    This function calculates the softmax cross entropy loss
    (https://cs231n.github.io/linear-classify/#softmax-classifier)between a given two-dimensional input vector `x` and a
    one-dimensional int-vector `t` that represents the groundtruth labels for classification.
    :param x: the input vector that is to be used to calculate the loss
    :param t: the groundtruth labels for the batch
    :return: the cross entropy loss of input and groundtruth labels
    """
    return SoftmaxCrossEntropy()(x, t)
