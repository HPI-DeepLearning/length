import numpy as np

from length import constants
from length.function import Function


class MeanSquaredError(Function):
    """
    This function calculates the Mean Squared Error between two given vectors, as described in
    https://en.wikipedia.org/wiki/Mean_squared_error
    """
    name = "MeanSquaredError"

    def __init__(self):
        super().__init__()
        self.difference = None

    @staticmethod
    def create_one_hot(data, shape):
        assert len(shape) == 2, "Providing integers as second input to MSE only works with two dimensional input vectors"
        data_container = np.zeros(shape, dtype=constants.DTYPE)
        data_container[np.arange(len(data)), data] = 1
        return data_container

    def internal_forward(self, inputs):
        x1, x2 = inputs

        if np.issubdtype(x2.dtype, np.integer):
            x2 = self.create_one_hot(x2, x1.shape)

        self.difference = x1 - x2
        squared_sum = np.sum(np.square(self.difference))
        return (squared_sum / self.difference.size).astype(constants.DTYPE),

    def internal_backward(self, inputs, gradients):
        x1, x2 = inputs
        gx, = gradients
        derived_value = 2 / x1.size * self.difference
        gradient = derived_value * gx

        if np.issubdtype(x2.dtype, np.integer):
            # in case we used MSE as loss function, we won't propagate any gradients to the loss
            return gradient, None

        return gradient, -gradient


def mean_squared_error(input_1, input_2):
    """
    This function calculates the Mean Squared Error between input_1 and input_2. Both inputs should be vectors of the
    same shape. You can also supply a one-dimensional list of integers.
    If you do so this vector will be converted to a one_hot representation that fits to the shape of the second
    input
    :param input_1: the first vector of any shape
    :param input_2: the second vector. Needs to have the same shape as the first vector, or be a one-dimensional int vector
    :return: the mean squared error between input_1 and input_2
    """
    return MeanSquaredError()(input_1, input_2)
