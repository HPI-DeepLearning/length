from length.function import Function


class Layer(Function):
    """
    Abstract Layer is a super class for all neural network layers. A layer behaves like a function, but also
    has to keep track of internal parameters (like weights and biases) that need to be optimized.
    """
    needs_optimizer = True
    name = "Layer"

    def __init__(self):
        super().__init__()
        self.optimizer = None

    def internal_update(self, parameter_deltas):
        """
        :param parameter_deltas: contains the delta of the parameters to be applied to each parameter
        (same structure as in internal_backward but without the first element)
        """
        raise NotImplementedError

    def backward(self, gradients):
        gradients = super().backward(gradients)
        input_gradient = gradients[:len(self.inputs)]
        parameter_gradients = gradients[len(self.inputs):]
        if len(parameter_gradients) > 0:
            parameter_deltas = self.optimizer.run_update_rule(parameter_gradients, self)
            self.internal_update(parameter_deltas)
        return input_gradient
