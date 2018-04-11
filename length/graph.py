import numpy as np

from length import constants


class Graph:
    """
    Graph stores data and computational history of the neural network
    """
    def __init__(self, data, predecessors=None, creator=None, name=None):
        self.predecessors = predecessors
        self.data = data
        self.creator = creator
        self.grad = None
        self.name = name

    def backward(self, optimizer):
        if self.creator is None:
            # early exit if we are at the top of the computational graph
            return

        # if the size of the data array is 1, we are at the bottom of the computational graph
        # so, we are starting with a gradient of 1
        if self.data.size == 1 and self.grad is None:
            self.grad = np.ones((1,), dtype=constants.DTYPE)

        candidate_layers = []
        seen_layers = set()

        def add_candidate_layer(candidate):
            if candidate is not None and candidate not in seen_layers:
                candidate_layers.append(candidate)
                seen_layers.add(candidate)

        add_candidate_layer(self)

        while candidate_layers:
            candidate_layer = candidate_layers.pop()
            if candidate_layer.creator is None:
                continue

            if candidate_layer.creator.needs_optimizer:
                candidate_layer.creator.optimizer = optimizer

            gradients = candidate_layer.creator.backward(candidate_layer.grad)

            for predecessor, gradient in zip(candidate_layer.predecessors, gradients):
                predecessor.grad = gradient
                if gradient is not None:
                    # the gradient flows to another layer (does not happen with loss layers)
                    add_candidate_layer(predecessor)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        name = getattr(self.creator, 'name', 'input') if self.name is None else self.name
        return "{} {}".format(name, self.shape)

    def visualize(self):
        print(self._visualization_as_str())

    def _visualization_as_str(self):
        layers = [self]

        i = 0
        prev_id = {self: -1}
        while i < len(layers):
            layer = layers[i]
            i += 1
            if layer.creator is None:
                continue

            for predecessor in layer.predecessors:
                if predecessor not in layers:
                    layers.append(predecessor)
                    prev_id[predecessor] = i - 1

        table = [(i - j, repr(layer), i - prev_id[layer]) for j, layer in enumerate(layers)]
        # append header last, since we print in reversed order
        table.append(("id", "layer", "next"))

        widths = [str(max(len(str(row[column])) for row in table)) for column in range(3)]
        template = "{:>" + widths[0] + "} | {:<" + widths[1] + "} | {:" + widths[2] + "}"

        lines = [template.format(*row) for row in table]
        return "\n".join(reversed(lines))
