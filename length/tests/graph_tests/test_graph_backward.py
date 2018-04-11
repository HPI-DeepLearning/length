import numpy as np

import length.functions as F

from length import constants
from length.graph import Graph
from length.layers import FullyConnected
from length.optimizers.sgd import SGD
from length.tests import gradient_checker


def test_graph_backward_no_layers():
    data = np.array([2], dtype=constants.DTYPE)
    data_graph = Graph(data)
    data_graph.backward(None)
    assert data_graph.grad is None


def test_graph_backward_only_functions_in_graph():
    data = np.array([2], dtype=constants.DTYPE)

    data_graph_1 = Graph(data)
    data_graph_2 = Graph(data)
    h = F.add(data_graph_1, Graph(data))
    h = F.add(h, data_graph_2)

    assert int(h.data) == 6

    h.backward(None)

    assert data_graph_1.grad == 1
    assert data_graph_2.grad == 1
    assert h.grad == 1

    h.grad = 2

    h.backward(None)
    assert data_graph_1.grad == 2
    assert data_graph_2.grad == 2


def test_graph_backward_with_layers():
    # use a fully connected layer and have a look whether the backward pass distributes the gradients correctly
    data = np.random.uniform(-1, 1, (2, 2)).astype(constants.DTYPE)
    labels = np.array([1, 1], dtype=np.int32)

    fc_layer = FullyConnected(2, 2)
    fc_layer.weights[...] = np.zeros_like(fc_layer.weights)
    fc_layer.bias[...] = np.array([-10, 10])

    def run_forward(inputs, labels):
        fc_result = fc_layer(inputs)
        loss = F.softmax_cross_entropy(fc_result, labels)
        return loss

    data_graph = Graph(data)
    label_graph = Graph(labels)

    loss = run_forward(data_graph, label_graph)

    optimizer = SGD(0.001)
    loss.backward(optimizer)

    assert label_graph.grad is None
    gradient_checker.assert_allclose(data_graph.grad, np.zeros_like(data_graph.grad))
    gradient_checker.assert_allclose(fc_layer.weights, np.zeros_like(fc_layer.weights))
    gradient_checker.assert_allclose(fc_layer.bias, np.array([-10, 10]))

    # change the labels and make sure that the gradients are different now
    # the absolute values of the gradients of one sample shall be higher than the gradients of the other sample
    labels = np.array([0, 1], dtype=np.int32)
    label_graph = Graph(labels)

    loss = run_forward(data_graph, label_graph)

    loss.backward(optimizer)

    assert label_graph.grad is None
    assert (np.abs(data_graph.grad[0]) > np.abs(data_graph.grad[1])).all()
