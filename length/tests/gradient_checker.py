import numpy as np


def compute_numerical_gradient(function, inputs, last_gradients, eps=1e-3):
    """
    Function that computes numerical gradient (difference quotient explanation:https://en.wikipedia.org/wiki/Difference_quotient)
    for a given input. This function is to be used to check the backward implementation of each function/layer
    :param function: a python function that takes no arguments and performs a forward pass of the function/layer in question
    :param inputs: a tuple of input arrays that is used to perform the forward pass of the function/layer
    :param last_gradients: the gradients that are to be passed to the function/layer during backward
    :param eps: epsilon to be used for calculation of numerical gradient
    :return: the numerical gradient of the function
    """
    gradients = tuple(np.zeros_like(x) for x in inputs)

    for x, gx in zip(inputs, gradients):
        flat_x = x.reshape(-1)
        flat_gx = gx.reshape(-1)

        for i in range(flat_x.size):
            original_value = flat_x[i]
            flat_x[i] = original_value - eps
            outputs_1 = function()
            flat_x[i] = original_value + eps
            outputs_2 = function()
            flat_x[i] = original_value

            for y1, y2, gy in zip(outputs_1, outputs_2, last_gradients):
                if gy is not None:
                    # calculate the difference quotient
                    numerator = float(sum(((y2 - y1) * gy).reshape(-1)))
                    denominator = 2 * eps
                    grad = numerator / denominator
                    flat_gx[i] += grad

    return gradients


def assert_allclose(actual, desired, atol=1e-5, rtol=1e-4, verbose=True):
    """
    Wrapper around numpy allclose assertion that uses better tolerance values for neural network computation
    :param actual: the first value to check
    :param desired: the second value to check
    :param atol: absolute allowed tolerance of values
    :param rtol: realtive allowed tolerance of values
    :param verbose: verbose output of allclose check
    """
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, verbose=verbose)
