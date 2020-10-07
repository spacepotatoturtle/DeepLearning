from Network import Network
from Image import Image


network = Network(28 ** 2, 10, 16, 2)
network.load("network.json")

# Commented-out code is what I used to train the network, and the rest is me testing the network.

# with open("Datasets/mnist_train.csv", "rt", newline='') as f:
#     r = csv.reader(f)
#
#     for j in range(600):
#
#         inputs_set = []
#         outputs_set = []
#
#         for i in range(100 * j, 100 + (100 * j)):
#             data = next(r)
#             inputs = []
#             for k in range(1, len(data)):
#                 inputs.append(int(data[k]))
#             inputs_set.append(inputs)
#             outputs = []
#             for k in range(10):
#                 if k == int(data[0]):
#                     outputs.append(1)
#                 else:
#                     outputs.append(0)
#             outputs_set.append(outputs)
#
#         network.optimize(inputs_set, outputs_set)
#         print("Set " + str(j) + " finished.")
# network.save("network.json")

test = Image("Datasets/mnist_test.csv", 1)
test.display()
network.run(test.inputs, telemetry=True, reset=True, target=test.outputs)

