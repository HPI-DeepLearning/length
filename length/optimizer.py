

class Optimizer:
    """
    An optimizer applies a certain update rule on given gradients and returns the parameter deltas
    """

    def run_update_rule(self, gradients, layer):
        """
        Does the actual optimization step and calculates the update
        :param gradients: the gradients of the layer that are to be used to optimize the parameters (a tuple of
        numpy arrays)
        :param layer: the layer to which the gradients belong
        :return: the deltas that are to be applied to the parameters
        """
        raise NotImplementedError
