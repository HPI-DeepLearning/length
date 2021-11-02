import numpy as np

from length.data_set import Batch
from length.models import MLP


def test_graph_backward_no_layers():
    model = MLP()
    model.forward(Batch(
        np.random.random((10, 784)),
        np.random.randint(0, 10, 10)
    ))
    string = model.loss._visualization_as_str()

    assert "id" in string
    assert "layer" in string
    assert "next" in string

    def assert_all_included(layer):
        if layer.creator is None:
            return
        assert repr(layer) in string
        for predecessor in layer.predecessors:
            assert_all_included(predecessor)

    assert_all_included(model.loss)

    column_lengths = [[len(column) for column in line.split("|")] for line in string.split("\n")]
    first = column_lengths[0]
    for other in column_lengths[1:]:
        for x, y in zip(first, other):
            assert x == y
