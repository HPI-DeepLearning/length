from length.graph import Graph


class Function:
    """
    Function is the base class for all computations of the neural network
    a function does not have any parameters that need to be optimized
    """
    needs_optimizer = False
    name = "Function"

    def __init__(self):
        self.inputs = None
        self.outputs = None

    def internal_forward(self, inputs):
        """
        inputs is a tuple of numpy arrays
        :returns outputs as tuple of numpy arrays
        """
        raise NotImplementedError

    def internal_backward(self, inputs, gradients):
        """
        :param inputs: is a tuple of numpy arrays
        :param gradients: is a tuple of numpy arrays
        :returns gradients as tuple of numpy arrays (first elements are gradients with respect to the inputs)
        """
        raise NotImplementedError

    def forward(self, graphs):
        requirement = "The input to forward must be a list/tuple which only includes Graph objects."
        assert isinstance(graphs, (list, tuple)), requirement
        assert all(isinstance(graph, Graph) for graph in graphs), requirement

        self.inputs = tuple(graph.data for graph in graphs)
        self.outputs = self.internal_forward(self.inputs)

        output_graphs = [Graph(output, predecessors=graphs, creator=self) for output in self.outputs]
        if len(output_graphs) == 1:
            return output_graphs[0]
        return output_graphs

    def backward(self, gradients):
        gradients = self.internal_backward(self.inputs, (gradients,))
        return gradients

    def __call__(self, *args):
        return self.forward(args)
