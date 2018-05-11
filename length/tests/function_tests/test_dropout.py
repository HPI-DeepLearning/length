import math

import numpy as np
import pytest

from length import constants
from length.functions import dropout
from length.functions.dropout import Dropout
from length.graph import Graph
from length.tests import gradient_checker
from length.tests.utils import retry


def _dropout(input, function):
    return input * function.mask


def get_data():
    data = np.random.random((4, 5)).astype(dtype=constants.DTYPE)
    return data


def test_dropout_forward_ratio_0():
    data = get_data()

    result = dropout(Graph(data), dropout_ratio=0.)
    gradient_checker.assert_allclose(result.data, data)


def test_dropout_forward_ratio_1():
    data = get_data()

    with pytest.raises(ValueError):
        dropout(Graph(data), dropout_ratio=1.)


def test_dropout_forward_ratio_0_5():
    data = get_data()

    # set the seed to get reliable results
    np.random.seed(2)

    result = dropout(Graph(data), dropout_ratio=0.5)
    # check that ca. 50% of all data points are zero now
    non_zero_elements = result.data.nonzero()
    assert math.isclose(data.size // 2, non_zero_elements[0].size, abs_tol=data.size * 0.1)

    # reset the seed in case this might matter
    np.random.seed()


@retry(3)
def test_dropout_backward():
    data = get_data()
    gradient = np.random.random(data.shape).astype(constants.DTYPE)

    data_graph = Graph(data)
    dropout_function = Dropout(0.5)
    dropout_result = dropout_function(data_graph)
    computed_gradients, = dropout_function.backward(gradient)

    f = lambda: _dropout(data, dropout_result.creator)
    numerical_gradients, = gradient_checker.compute_numerical_gradient(f, (data,), gradient, eps=0.1)

    gradient_checker.assert_allclose(computed_gradients, numerical_gradients)


def test_ropout_test_mode():
    data = get_data()

    dropout_result = dropout(data, dropout_ratio=0.5, train=False)
    # there should be no changes in test mode
    gradient_checker.assert_allclose(dropout_result.data, data)
