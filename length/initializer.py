

class Initializer:
    """
    Base structure for all initializers
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, array):
        """
        initialize the given array according to the initialization rule implemented in this initializer
        :param array: the array to initialize
        :return: the initialized array
        """
        raise NotImplementedError
