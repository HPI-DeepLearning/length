import numpy as np

from length.data_sets import MNIST, FashionMNIST
from length.models import MLP
from length.optimizers import Adam, SGD


def test_train():
    np.seterr(divide='raise')

    for optimizer in [SGD(0.001), Adam(0.001)]:
        for data_set in [MNIST(64), FashionMNIST(64)]:
            for i in range(10):
                model = MLP()

                for iteration, batch in enumerate(data_set.train):
                    model.forward(batch)
                    model.backward(optimizer)
                    break
