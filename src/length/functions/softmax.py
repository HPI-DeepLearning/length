import numpy as np

from length.function import Function


def log_softmax(x):
    def logsumexp(x):
        m = x.max(axis=1, keepdims=True)
        y = x - m
        np.exp(y, out=y)
        s = y.sum(axis=1, keepdims=True)
        np.log(s, out=s)
        m += s
        return m

    log_z = logsumexp(x)
    # log(e^(x)/log_z)
    y = x - log_z
    return y


class Softmax(Function):
    name = "Softmax"

    def __init__(self):
        super().__init__()
        self.y = None

    def internal_forward(self, inputs):
        x, = inputs
        assert x.ndim == 2, "Softmax only supports two-dimensional input"
        self.y = np.exp(log_softmax(x))
        return self.y,

    def internal_backward(self, inputs, gradients):
        x, = inputs
        grad_in, = gradients
        grad_x = self.y * grad_in
        sum_grad_x = grad_x.sum(axis=1, keepdims=True)
        grad_x -= self.y * sum_grad_x

        assert x.shape == grad_x.shape

        return grad_x,


def softmax(x):
    """
    The softmax function takes a two-dimensional input (of shape BxN) and computes the softmax function
    (https://en.wikipedia.org/wiki/Softmax_function)on the last dimension of this input. The result is a two-dimensional
    vector (of shape BxN) that contains a probability distribution for each sample b of the batch with batch size B.
    :param x: the two-dimensional input vector to calculate the softmax on
    :return: a two-dimensional vector with softmax applied to the last dimension
    """
    return Softmax()(x)
