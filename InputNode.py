from Node import Node


class InputNode(Node):

    """
    Hidden Node, except no activation function.
    """

    def fire(self):
        self.input_sum += self.bias

        self.output_value = self.input_sum

        for connection in self.outConnections:
            connection.toNode.input_sum += self.output_value * connection.weight

    def type(self):
        return 'n'
