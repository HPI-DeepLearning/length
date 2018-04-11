import numpy as np

from length.function import Function
from length.functions.softmax import Softmax


class Accuracy(Function):
    name = "Accuracy"

    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def internal_forward(self, inputs):
        x, t = inputs
        softmaxed_x, = self.softmax.internal_forward((x,))
        predicted_classes = np.argmax(softmaxed_x, axis=1)

        num_correct = (predicted_classes == t).sum()

        return num_correct / len(t),

    def internal_backward(self, inputs, gradients):
        return None, None


def accuracy(x, t):
    """
    The accuracy function takes the output of a classifier (i.e. the last fully connected layer) and also an int-vector
    with groundtruth labels and calculates the accuracy of this prediction. Please Note: This function can not be used
    for backpropagation!
    :param x: a two-dimensional (Shape: (B,N) with B being the batch_size) vector that represents the classification
    result of the network.
    :param t: a one-dimensional vector of ints that represents the groundtruth labels
    :return: the accuracy of the classification.
    """
    return Accuracy()(x, t)
