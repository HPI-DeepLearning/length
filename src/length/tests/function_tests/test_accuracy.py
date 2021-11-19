import numpy as np

import length.functions as F
from length.graph import Graph
from length.tests.utils import init


def calc_accuracy(data, labels):
    accuracy = F.accuracy(Graph(data), Graph(labels))
    return accuracy


def get_base_data():
    first_element = init([-10, -10, -5, -4, 5, -1])
    second_element = init([-10, -10, 7, 3, 4, -1])
    data = np.stack((first_element, second_element))
    labels = np.array([4, 0])
    return data, labels


def test_accuracy_forward():
    data, labels = get_base_data()

    accuracy = calc_accuracy(data, labels)
    assert accuracy.data == 0.5

    labels = np.array([4, 2])
    accuracy = calc_accuracy(data, labels)
    assert accuracy.data == 1


def test_accuracy_backward():
    data, labels = [Graph(x) for x in get_base_data()]

    accuracy = F.accuracy(data, labels)
    accuracy.backward(None)

    assert data.grad is None
    assert labels.grad is None
