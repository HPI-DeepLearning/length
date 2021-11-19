import numpy as np

from length import constants
from length.graph import Graph
from length.layers import FullyConnected
from length.tests import gradient_checker
from length.tests.utils import init, retry


def fixed_case():
    data = init([
        [0.44,  0.06, 0.33,  0.76],
        [0.53,  0.65, 0.06, -0.35],
        [0.29, -0.90, 0.86,  0.76],
    ])

    weights = init([
        [-0.74, -0.44],
        [-0.51,  0.63],
        [ 0.73, -0.38],
        [ 0.24, -0.43],
    ])

    bias = init([0.86, 0.63])

    expected = init([
        [0.9271,  0.022],
        [0.0961,  0.934],
        [1.9146, -0.7182],
    ])

    return data, weights, bias, expected


def test_fully_connected_forward():
    data, weights, bias, expected = fixed_case()

    layer = FullyConnected(4, 2)
    layer.weights = weights
    layer.bias = bias

    layer_output = layer(Graph(data))
    gradient_checker.assert_allclose(layer_output.data, expected)


@retry(3)
def test_fully_connected_backward():
    data = np.random.uniform(-1, 1, (10, 50)).astype(constants.DTYPE)

    gradient = np.full((10, 20), 2, dtype=constants.DTYPE)

    layer = FullyConnected(50, 20)
    comp_grad_x, comp_grad_weight, comp_grad_bias = layer.internal_backward((data,), (gradient,))

    f = lambda: layer.internal_forward((data,))
    num_grad_x, num_grad_weight, num_grad_bias = gradient_checker.compute_numerical_gradient(f, (data, layer._weights, layer.bias), (gradient,), eps=1e-2)

    gradient_checker.assert_allclose(comp_grad_x, num_grad_x, atol=1e-4)
    gradient_checker.assert_allclose(comp_grad_weight, num_grad_weight, atol=1e-4)
    gradient_checker.assert_allclose(comp_grad_bias, num_grad_bias, atol=1e-4)


def test_initialization():
    layer = FullyConnected(50, 42)
    assert not (layer.weights == 0).all()
