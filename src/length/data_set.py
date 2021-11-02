from length.graph import Graph


class Batch:
    def __init__(self, data, labels):
        self.data = Graph(data, name="input[data]")
        self.labels = Graph(labels, name="input[labels]")


class DataSet:
    """
    Abstract DataSet, it should implement ways to retrieve test and train data and support shuffling
    """
    def __init__(self, batch_size, shuffle=True, repeat=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat

    @property
    def test(self):
        """
        :return: an iterator over the test set.
        """
        raise NotImplementedError()

    @property
    def train(self):
        """
        :return: an iterator over the train set.
        """
        raise NotImplementedError()
