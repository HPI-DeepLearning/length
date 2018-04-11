import math

import numpy as np

from length import constants
from length.functions import softmax_cross_entropy
from length.functions.softmax_cross_entropy import SoftmaxCrossEntropy
from length.graph import Graph
from length.tests import gradient_checker
from length.tests.utils import init, retry


def get_data():
    data = np.random.uniform(-1, 1, (10, 20)).astype(constants.DTYPE)
    labels = np.random.randint(0, 20, (10,)).astype(np.int32)
    return data, labels


def test_softmax_cross_entropy_forward():
    data, labels = get_data()

    softmax_loss = softmax_cross_entropy(Graph(data), Graph(labels))

    y = np.exp(data)
    expected_loss = 0
    for i in range(len(y)):
        expected_loss -= math.log(y[i, labels[i]] / y[i].sum())
    expected_loss /= len(y)

    assert math.isclose(float(softmax_loss.data), expected_loss, rel_tol=1e-4, abs_tol=1e-5)


@retry(3)
def test_softmax_cross_entropy_backward():
    data, labels = get_data()
    gradient = init([2])

    loss_function = SoftmaxCrossEntropy()
    loss_function(Graph(data), Graph(labels))
    computed_gradient_data, computed_gradient_label = loss_function.backward(gradient)
    assert computed_gradient_label is None

    f = lambda: loss_function.internal_forward((data, labels))
    numerical_gradient_data, _ = gradient_checker.compute_numerical_gradient(f, (data, labels), (gradient,), eps=1e-2)

    gradient_checker.assert_allclose(computed_gradient_data, numerical_gradient_data, atol=1e-4)
