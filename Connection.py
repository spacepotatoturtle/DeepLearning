class Connection:

    def __init__(self, fnode, tnode, wt):
        # Gradient is the partial derivative of cost with respect to this weight for the current MNIST image.
        # To get the average gradient over a training batch, weight_adj is the sum of the gradients
        # and adj_num is the number of gradients summed.
        # Both are reset to 0 by the network class after a training batch has been completed.
        self.weight_adj = 0
        self.gradient = 0
        self.adj_num = 0
        self.fromNode = fnode
        self.toNode = tnode
        self.weight = wt

    # adjust_weight() doesn't actually change anything. It calculates the gradient and changes the weight_adj
    # and adj_num in turn. implement_adj() actually applies the changes to the connection's weight, to be called
    # only at the end of a training batch.

    def adjust_weight(self):
        self.gradient = self.toNode.gradient * self.fromNode.output_value
        self.weight_adj -= self.gradient
        self.adj_num += 1

    def implement_adjust(self):
        self.weight += float(self.weight_adj / self.adj_num)
        self.reset_adjust()

    # I don't know why I didn't just take this method and put it in the implement_adjust method.
    def reset_adjust(self):
        self.weight_adj = 0
        self.adj_num = 0
