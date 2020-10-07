from Node import Node


class OutputNode(Node):

    """
    I'm supposed to be using a softmax activation function for output nodes, but I chose
    not to, mostly because softmax's total derivative for each individual node included values from all
    the other output nodes, and I didn't like that.
    """

    def adjust_bias(self, target=None):
        if target is None:
            print("Error: no target output to train off of.")
        # Since output nodes are the first in the backprop process, they need to calculate gradient based
        # on the target outputs.
        self.gradient = self.d_sigmoid(self.input_sum) * 2 * (self.output_value - target)
        self.bias_adj -= self.gradient
        self.adj_num += 1

    def type(self):
        return 'o'
