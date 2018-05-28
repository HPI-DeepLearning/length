import numpy as np

from length.function import Function


class Dropout(Function):
    name = "Dropout"

    def __init__(self, dropout_ratio):
        super().__init__()
        if not 0.0 <= dropout_ratio < 1:
            raise ValueError("dropout_ratio must be in range [0, 1)")
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def internal_forward(self, inputs):
        x, = inputs
        scale = x.dtype.type(1. / (1 - self.dropout_ratio))
        flag = np.random.rand(*x.shape) >= self.dropout_ratio
        self.mask = scale * flag
        return x * self.mask,

    def internal_backward(self, inputs, gradients):
        gradient, = gradients
        return gradient * self.mask,


def dropout(x, dropout_ratio=0.5, train=True):
    """
    This function implements dropout (http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf),
    a regularization method for neural networks.
    :param x: the input vector where parts shall be dropped
    :param dropout_ratio: the ratio of which to perform dropout
    :param train: whether we are currently running in train or testing mode (default: True)
    :return: a vector with a portion of elements zeroed out, this portion is defined by `dropout_ratio`.
    """
    if train:
        return Dropout(dropout_ratio)(x)
    return x
