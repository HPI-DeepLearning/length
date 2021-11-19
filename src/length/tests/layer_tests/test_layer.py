from length.layer import Layer
from length.layers import *


def test_layer_needs_optimizer():
    # here we test that each layers wants to have an optimizer
    for sub_class in Layer.__subclasses__():
        assert sub_class.needs_optimizer is True
