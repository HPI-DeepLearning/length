import pytest

from length.function import Function
from length.functions import *
from length.graph import Graph
from length.layer import Layer


def test_function_no_optimizer_necessary():
    # test that a function always says that it does not need an optimizer
    # in order to do that we dynamically get all subclasses of the class function and make sure to not
    # have any subclass of the class layer
    function_subclasses = set(Function.__subclasses__())
    layer_subclasses = set([Layer] + Layer.__subclasses__())
    function_subclasses = function_subclasses.difference(layer_subclasses)

    # we then create an instance of each class and check that it says it wants no optimizer
    for sub_class in function_subclasses:
        assert sub_class.needs_optimizer is False


def test_correct_wrong_input():
    with pytest.raises(AssertionError) as info:
        f = Function()
        f.forward(15)
    message = str(info.value)
    assert "must be a list/tuple" in message
    assert "only includes Graph objects" in message


def test_correct_wrong_input_2():
    with pytest.raises(AssertionError) as info:
        f = Function()
        f.forward([15])
    message = str(info.value)
    assert "must be a list/tuple" in message
    assert "only includes Graph objects" in message


def test_correct_input():
    # this should only raise a not implemented error as internal_forward is not implemented
    with pytest.raises(NotImplementedError):
        f = Function()
        f.forward([Graph(15)])
