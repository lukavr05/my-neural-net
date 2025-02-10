import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class Node():
    def __init__(self, w, b):
        self.weight = w
        self.bias = b

    def calculateOutput(self, input):
        output = numpy.dot(self.weight, input) + self.bias
        return sigmoid(output)


# weights = numpy.array([0, 1])
# bias = 4
# n = Node(weights, bias)
# print(n.calculateOutput(numpy.array([2, 3])))
