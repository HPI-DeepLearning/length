import numpy as np

from length.optimizer import Optimizer


class Adam(Optimizer):
    """
    The Adam optimizer (see https://arxiv.org/abs/1412.6980)
    :param learning_rate: initial step size
    :param beta1: Exponential decay rate of the first order moment
    :param beta2: Exponential decay rate of the second order moment
    :param eps: Small value for numerical stability
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # TODO: add more initialization code

    def run_update_rule(self, gradients, layer):
        # TODO: implement Adam update rule as specified in https://arxiv.org/abs/1412.6980
        param_deltas = [np.zeros_like(g) for g in gradients]
        return param_deltas
