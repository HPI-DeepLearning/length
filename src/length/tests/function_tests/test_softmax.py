import numpy as np

from length import constants
from length.functions.softmax import Softmax, softmax
from length.graph import Graph
from length.tests import gradient_checker
from length.tests.utils import retry


def test_softmax_forward():
    data = np.random.uniform(-1, 1, (3, 10)).astype(constants.DTYPE)

    output = softmax(Graph(data)).data

    expected_output = np.exp(data)
    for i in range(len(output)):
        expected_output[i] /= expected_output[i].sum()

    gradient_checker.assert_allclose(output, expected_output)


@retry(3)
def test_softmax_backward():
    data = np.random.uniform(-1, 1, (3, 10)).astype(constants.DTYPE)
    gradient = np.random.uniform(-1, 1, (3, 10)).astype(constants.DTYPE)

    data_graph = Graph(data)
    softmax_function = Softmax()
    softmax_function(data_graph)
    computed_gradients, = softmax_function.backward(gradient)

    f = lambda: softmax_function.internal_forward((data,))
    numerical_gradients, = gradient_checker.compute_numerical_gradient(f, (data,), (gradient,), eps=1e-2)

    gradient_checker.assert_allclose(computed_gradients, numerical_gradients)
