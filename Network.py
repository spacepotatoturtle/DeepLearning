from Layer import Layer
import json


class Network:

    """
    Manages all of the nodes. I made the decision to keep the training process internal to the
    network class, instead of having an external training process, since all the variables I would need to
    access were more accessible here.
    Really, most of the work is done by the individual nodes, both in feedforward and backpropagation.
    The network's job is to make sure they are done in the right order.
    """

    def __init__(self, num_inputs, num_outputs, num_per_hidden, num_hidden):

        layers = []

        # Generate layers
        for layerNum in range(num_hidden + 2):
            layers.append(Layer(layerNum, []))

        node_num = 0

        # Generate nodes

        for i in range(num_inputs):
            layers[0].add_node(node_num, 'n')
            node_num += 1

        for i in range(num_outputs):
            layers[-1].add_node(node_num, 'o')
            node_num += 1

        for j in range(num_hidden):
            for i in range(num_per_hidden):
                layers[1 + j].add_node(node_num)
                node_num += 1

        # Generate connections

        for layer in layers:
            for node in layer.nodes:
                if layer.number != len(layers) - 1:
                    for to_node in layers[layer.number + 1].nodes:
                        node.connect_to(to_node)

        # I should have made it self.layers from the beginning
        self.layers = layers

    def run(self, inputs, **kwargs):
        # Fires the nodes of each layer in sequence.
        if len(inputs) == self.layers[0].size():
            for layer in self.layers:
                if layer.number != 0:  # Also sets the initial value for input nodes.
                    for node in layer.nodes:
                        node.fire()
                else:
                    for node in layer.nodes:
                        node.input_sum = inputs[layer.nodes.index(node)]
                        node.fire()
            output_values = []
            # A whole bunch of extra attributes: print outputs, reset input/output values of each node,
            # and error calculation (the cost function).
            for node in self.layers[-1].nodes:
                output_values.append(node.output_value)
                if "telemetry" in kwargs:
                    if kwargs["telemetry"]:
                        print("Output " + node.name + ": " + str(node.output_value))
            if "reset" in kwargs:
                if kwargs["reset"]:
                    self.reset()
            if "target" in kwargs:
                error = 0
                if len(kwargs["target"]) == len(output_values):
                    for i in range(len(kwargs["target"])):
                        error += (output_values[i] - kwargs["target"][i]) ** 2
                    print("Cost: " + str(error))
                else:
                    print("Error: size of network outputs does not match size of target outputs.")
            return output_values
        else:
            print("Error: size of inputs do not match size of input nodes.")

    def cost(self, inputs, target_outputs):
        net_outputs = self.run(inputs)
        error = 0
        if len(target_outputs) == len(net_outputs):
            for i in range(len(target_outputs)):
                error += (net_outputs[i] - target_outputs[i]) ** 2
        else:
            print("Error: size of network outputs does not match size of target outputs.")
        self.reset()
        return error

    # This method calculates gradients for a single image file. The gradients are then recorded in the
    # bias_adj, adj_num of the node instances and the weight_adj, adj_num of the connection instances (see
    # those files for explanations of what the variables are for). It does not implement the changes, and thus
    # does not actually change the network in any way.
    def adjust(self, inputs, target_outputs):
        # A really clunky-looking backprop system.
        # The first section is specifically for the output nodes, because they need the target outputs.
        self.run(inputs, target=target_outputs)
        for i in range(len(self.layers[-1].nodes)):
            self.layers[-1].nodes[i].adjust_bias(target_outputs[i])

        # This section is everything else. First the connections going out from a layer, then the layer's
        # nodes themselves.
        for layer in [x for x in list(reversed(self.layers)) if list(reversed(self.layers)).index(x) > 0]:
            for node in layer.nodes:
                for connection in node.outConnections:
                    connection.adjust_weight()
            for node in layer.nodes:
                node.adjust_bias()

        # Clears network after all the input/output values have been used for calculations.
        self.reset()

    # This method runs the adjust method for an entire batch of training images, and then actually
    # alters the network accordingly by implementing the averaged gradients of each bias and weight.
    def optimize(self, inputs_set, outputs_set):
        if len(inputs_set) == len(outputs_set):
            for i in range(len(inputs_set)):
                self.adjust(inputs_set[i], outputs_set[i])

            for layer in self.layers:
                for node in layer.nodes:
                    node.implement_adjust()
                    for connection in node.outConnections:
                        connection.implement_adjust()
        else:
            print("Error: input set and output set do not match in size.")

    def find_node(self, number):
        for layer in self.layers:
            for node in layer.nodes:
                if node.number == number:
                    return node
        return None

    def reset(self):
        for layer in self.layers:
            for node in layer.nodes:
                node.reset()

    def clear(self):
        del self.layers
        self.layers = []

    # A very rudimentary way to encode all the nodes, biases, weights, and overall layer structure
    # into a json file using a cascade of lists.
    #
    #  layer_data  node_data                                   connection_data
    # [[layer_num, [node_num, node_name, node_type, node_bias, [connects_to_node_num, weight], [...], ...], ...], ...]

    def save(self, file):
        data = []
        for layer in self.layers:
            # This is nearly the only time layer and node ID numbers are useful.
            layer_data = [layer.number]
            for node in layer.nodes:
                node_data = [node.number, node.name, node.type(), node.bias]
                for connection in node.outConnections:
                    connection_data = [connection.toNode.number, connection.weight]
                    node_data.append(connection_data)
                layer_data.append(node_data)
            data.append(layer_data)
        with open(file, "w") as f:
            f.truncate(0)
            json.dump(data, f, indent=2)

    # And, to decode it.

    def load(self, file):
        self.clear()
        with open(file) as f:
            data = json.load(f)

            # First add all the nodes
            for layer_data in data:
                self.layers.append(Layer(layer_data[0], []))
                for node_data in [x for x in layer_data if layer_data.index(x) > 0]:
                    self.layers[-1].add_node(node_data[0], node_data[2], node_data[3], label=node_data[1])

            # Then add all the connections
            for layer_data in data:
                for node_data in [x for x in layer_data if layer_data.index(x) > 0]:
                    for connection_data in [x for x in node_data if node_data.index(x) > 3]:
                        self.find_node(node_data[0]).connect_to(self.find_node(connection_data[0]), connection_data[1])
