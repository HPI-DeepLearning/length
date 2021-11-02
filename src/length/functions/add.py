from length.function import Function


class Add(Function):
    name = "Add"

    def internal_forward(self, inputs):
        x, y = inputs
        return x + y

    def internal_backward(self, inputs, gradients):
        gradient, = gradients
        return gradient, gradient


def add(x, y):
    """
    This function performs element-wise addition of two input vectors. Please Note: Both vectors must have the same
    shape!
    :param x: the first vector where each element shall be summed with the elements of the second vector
    :param y: the second vector
    :return: one vector containing the element-wise sum of x and y
    """
    return Add()(x, y)
