from length.layers import FullyConnected
import length.functions as F


class MLP:
    """
    Simple network with two fully connected layers
    """
    def __init__(self):
        self.fully_connected_1 = FullyConnected(784, 512)
        self.fully_connected_2 = FullyConnected(512, 512)
        self.fully_connected_3 = FullyConnected(512, 10)

        self.loss = None
        self.predictions = None

    def forward(self, batch, train=True):
        """
        This runs the forward pass with the model
        :param batch: the batch which should be forwarded
        """
        hidden = self.fully_connected_1(batch.data)
        hidden = self.fully_connected_2(hidden)
        self.predictions = self.fully_connected_3(hidden)
        self.loss = F.mean_squared_error(self.predictions, batch.labels)

    def backward(self, optimizer):
        """
        This runs the backward pass (should *not* be used in test loop)
        :param optimizer: the optimizer which calculated the updates
        """
        self.loss.backward(optimizer)
