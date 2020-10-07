import numpy as np
import decimal as dc
from Connection import Connection


class Node:

    def __init__(self, num, connections=None, bias=np.random.normal(0, 1), **kwargs):
        # Before activation function
        self.input_sum = 0
        # After
        self.output_value = 0
        # As with the connection class, the gradient is the partial derivative of cost with respect to this
        # node's bias. The other two variables are for getting the average gradient over an entire training
        # batch. The bias_adj is the sum of the gradients, and adj_num is the number of gradients. Both
        # are reset by the network at the end of a training batch.
        self.bias_adj = 0
        self.gradient = 0
        self.adj_num = 0

        if connections is None:
            connections = []
        self.number = num
        self.outConnections = connections
        self.bias = bias
        if "label" in kwargs:
            self.name = kwargs["label"]
        else:
            self.name = str(num)

    def fire(self):
        self.input_sum += self.bias

        self.output_value = self.sigmoid(self.input_sum)

        # This is much more oriented towards a NEAT-style connection system, where connections between
        # nodes are a lot more fluid. Take advantage of the predictability and construct matrices between
        # adjacent layers. MAY DO: Overhaul connection system.

        for connection in self.outConnections:
            connection.toNode.input_sum += self.output_value * connection.weight

    def reset(self):
        self.input_sum = 0
        self.output_value = 0

    # The default node type is hidden (h).
    def type(self):
        return 'h'

    # The to_node is the node that this node fires signals at.
    def connect_to(self, to_node, weight=None):
        if weight is None:
            self.outConnections.append(Connection(self, to_node, np.random.normal(0, 1)))
        else:
            self.outConnections.append(Connection(self, to_node, weight))

    def find_connection(self, to_node):
        for connection in self.outConnections:
            if connection.toNode == to_node:
                return connection

    # adjust_bias() doesn't actually change anything. It calculates the gradient and changes the bias_adj
    # and adj_num in turn. implement_adj() actually applies the changes to the node's bias, to be called
    # only at the end of a training batch.

    def adjust_bias(self):
        self.gradient = self.d_sigmoid(self.input_sum)
        for connection in self.outConnections:
            self.gradient = self.gradient * connection.weight * connection.toNode.gradient
        self.bias_adj -= self.gradient
        self.adj_num += 1

    def implement_adjust(self):
        self.bias += self.bias_adj / self.adj_num
        self.reset_adjust()

    # I don't know why I didn't just take this method and put it in the implement_adjust method.
    def reset_adjust(self):
        self.bias_adj = 0
        self.adj_num = 0

    # Activation function things

    @staticmethod
    def sigmoid(x):
        dc.getcontext().prec = 28
        y = float(1 / (1 + dc.Decimal(np.e) ** dc.Decimal(-1 * x)))
        # y = 1 / (1 + np.e ** (-1 * x))
        return y

    def d_sigmoid(self, x):
        y = self.sigmoid(x) * (1 - self.sigmoid(x))
        return y
