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
        param_deltas = [self.lr * gradient for gradient in gradients]
        return param_deltas
