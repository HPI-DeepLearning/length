import numpy as np

from length import constants
from length.functions import relu
from length.functions.relu import Relu
from length.graph import Graph
from length.tests import gradient_checker
from length.tests.utils import init, retry


def test_relu_forward():
    data = init([
        [-0.620304  , -0.1285682 ,  0.4867715 ,  0.09824127],
        [-0.37919873, -0.9272095 , -0.0704312 ,  0.35593647],
        [ 0.19380952,  0.06425636,  0.21729442, -0.3168534 ],
        [-0.62586236, -0.4846    ,  0.84347826,  0.22025743],
        [ 0.02966821, -0.2127131 , -0.33760294, -0.9477733 ]
    ])

    desired = init([
        [0.        , 0.        , 0.4867715 , 0.09824127],
        [0.        , 0.        , 0.        , 0.35593647],
        [0.19380952, 0.06425636, 0.21729442, 0.        ],
        [0.        , 0.        , 0.84347826, 0.22025743],
        [0.02966821, 0.        , 0.        , 0.        ]
    ])

    relu_output = relu(Graph(data))
    np.testing.assert_allclose(relu_output.data, desired)


@retry(3)
def test_relu_backward():
    data = np.random.uniform(-1, 1, (5, 4)).astype(constants.DTYPE)
    gradient = np.random.random(data.shape).astype(dtype=constants.DTYPE)

    data_graph = Graph(data)
    relu_function = Relu()
    relu_function(data_graph)
    computed_gradients, = relu_function.backward(gradient)

    f = lambda: relu_function.internal_forward((data,))
    numerical_gradients, = gradient_checker.compute_numerical_gradient(f, (data,), (gradient,))

    gradient_checker.assert_allclose(computed_gradients, numerical_gradients)
