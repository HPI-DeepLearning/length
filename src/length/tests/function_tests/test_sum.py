import numpy as np

from length import constants
from length.functions.sum import Sum, sum
from length.graph import Graph
from length.tests import gradient_checker
from length.tests.utils import init, retry


def test_sum_forward():
    data = np.random.uniform(-1, 1, (3, 2)).astype(constants.DTYPE)

    sum_output = sum(Graph(data))
    np.testing.assert_allclose(sum_output.data, data.sum())


@retry(3)
def test_sum_backward():
    data = np.random.uniform(-1, 1, (3, 2)).astype(constants.DTYPE)
    gradient = init([2])

    data_graph = Graph(data)
    sum_function = Sum()
    sum_function(data_graph)
    computed_gradients, = sum_function.backward((gradient,))

    f = lambda: sum_function.internal_forward((data,))
    numerical_gradients, = gradient_checker.compute_numerical_gradient(f, (data,), (gradient,))

    gradient_checker.assert_allclose(computed_gradients, numerical_gradients, atol=1e-4, rtol=1e-3)
