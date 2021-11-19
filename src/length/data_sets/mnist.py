from .mnist_like import MNISTLike


class MNIST(MNISTLike):
    """
    The hand-written digit data set
    """

    name = "mnist"
    url = "http://yann.lecun.com/exdb/mnist/"
