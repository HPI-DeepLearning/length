import math

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

        # map from layer id to a list of numpy arrays
        self.m_values = {}
        self.v_values = {}
        # map from layer id to int (time step)
        self.t_values = {}

        self.current_id = -1
        self._initialized = set()

    def run_update_rule(self, gradients, layer):
        self.current_id = id(layer)

        if not self.initialized:
            self.initialize(gradients)

        self.t_values[self.current_id] += 1
        t = self.t_values[self.current_id]

        param_deltas = []

        for i, gradient in enumerate(gradients):
            m = self.m_values[self.current_id][i]
            v = self.v_values[self.current_id][i]

            m += (1 - self.beta1) * (gradient - m)
            v += (1 - self.beta2) * (gradient * gradient - v)

            m_fix = m / (1 - self.beta1 ** t)
            v_fix = v / (1 - self.beta2 ** t)

            param_deltas.append(self.learning_rate * m_fix / (np.sqrt(v_fix) + self.eps))
        return param_deltas

    def initialize(self, gradients):
        self.m_values[self.current_id] = [np.zeros_like(gradient) for gradient in gradients]
        self.v_values[self.current_id] = [np.zeros_like(gradient) for gradient in gradients]
        self.t_values[self.current_id] = 0
        self._initialized.add(self.current_id)

    @property
    def initialized(self):
        return self.current_id in self._initialized
