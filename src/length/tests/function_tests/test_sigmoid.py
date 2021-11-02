import numpy as np

from length import constants
from length.functions import sigmoid
from length.functions.sigmoid import Sigmoid
from length.graph import Graph
from length.tests import gradient_checker
from length.tests.utils import init, retry


def test_sigmoid_forward():
    data = init([
        [-0.22342056,  0.6927312 ],
        [ 0.4227562 , -0.59764487],
        [ 0.7870561 ,  0.372502  ]
    ])

    sigmoid_output = sigmoid(Graph(data))
    desired = init([
        [0.44437608, 0.66657424],
        [0.6041426 , 0.3548827 ],
        [0.6871989 , 0.5920634 ]
    ])

    np.testing.assert_allclose(sigmoid_output.data, desired)


@retry(3)
def test_sigmoid_backward():
    data = np.random.uniform(-1, 1, (3, 2)).astype(constants.DTYPE)
    gradient = np.random.random(data.shape).astype(constants.DTYPE)

    data_graph = Graph(data)
    sigmoid_function = Sigmoid()
    sigmoid_function(data_graph)
    computed_gradients, = sigmoid_function.backward(gradient)

    f = lambda: sigmoid_function.internal_forward((data,))
    numerical_gradients, = gradient_checker.compute_numerical_gradient(f, (data,), (gradient,))

    gradient_checker.assert_allclose(computed_gradients, numerical_gradients, atol=1e-4, rtol=1e-3)
