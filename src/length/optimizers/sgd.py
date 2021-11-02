from length.optimizer import Optimizer


class SGD(Optimizer):
    """
    An optimizer that does plain Stochastic Gradient Descent
    (https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Iterative_method)
    :param lr: the learning rate to use for optimization
    """

    def __init__(self, lr):
        self.lr = lr

    def run_update_rule(self, gradients, _):
        # TODO: implement SGD update rule
        # HINT: it should not be more than one line ;)
        param_deltas = None
        return param_deltas
