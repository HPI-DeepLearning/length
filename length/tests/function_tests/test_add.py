import numpy as np

from length.functions.add import add, Add
from length.graph import Graph
from length.tests import gradient_checker
from length.tests.utils import init, retry


def test_add_forward():
    data = init([2])

    result = add(Graph(data), Graph(data))
    assert result.data == 4


@retry(3)
def test_add_backward():
    data = init([2])
    gradient = init([1])

    add_function = Add()
    computed_gradients_1, computed_gradients_2 = add_function.backward(gradient)

    data_copy = np.copy(data)
    f = lambda: add_function.internal_forward((data, data_copy))
    numerical_gradients_1, numerical_gradients_2 = gradient_checker.compute_numerical_gradient(f, (data, data_copy), (gradient,))

    gradient_checker.assert_allclose(computed_gradients_1, numerical_gradients_1)
    gradient_checker.assert_allclose(computed_gradients_2, numerical_gradients_2)
