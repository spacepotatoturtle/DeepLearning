from Node import Node
from OutputNode import OutputNode
from InputNode import InputNode


class Layer:

    """
    Honestly, this class is not very necessary. The layer structure could have been easily created using
    a list of lists of nodes in the network class. The add_node() method basically just passes parameters down
    to the node __init__() method.
    """

    def __init__(self, no, nodes):
        self.nodes = nodes
        self.number = no

    def add_node(self, num, type='h', bias=0, connections=None, **kwargs):
        if connections is None:
            connections = []
        if type == 'h':
            if "label" in kwargs:
                self.nodes.append(Node(num, connections, bias, label=kwargs["label"]))
            else:
                self.nodes.append(Node(num, connections, bias))
        elif type == 'n':
            if "label" in kwargs:
                self.nodes.append(InputNode(num, connections, bias, label=kwargs["label"]))
            else:
                self.nodes.append(InputNode(num, connections, bias))
        elif type == 'o':
            if "label" in kwargs:
                self.nodes.append(OutputNode(num, connections, bias, label=kwargs["label"]))
            else:
                self.nodes.append(OutputNode(num, connections, bias))

    def size(self):
        return len(self.nodes)
