import numpy as np

from length.initializer import Initializer


class Xavier(Initializer):
    """
    Xavier initializer that initializes weights following the initialization scheme proposed in:
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi
    """

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def __call__(self, array):
        # TODO: overwrite values in array with a correctly initialized one
        array[...] = np.zeros_like(array)
