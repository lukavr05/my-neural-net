import numpy
import node


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class Node():
    def __init__(self, w, b):
        self.weight = w
        self.bias = b

    def calculateOutput(self, input):
        output = numpy.dot(self.weight, input) + self.bias
        return sigmoid(output)


class NeuralNetwork:
    def __init__(self):
        weight = numpy.array([0, 1])
        bias = 0

        self.h1 = node.Node(weight, bias)
        self.h2 = node.Node(weight, bias)
        self.o1 = node.Node(weight, bias)

    def feedforward(self, x):
        out_h1 = self.h1.calculateOutput(x)
        out_h2 = self.h2.calculateOutput(x)

        out_o1 = self.o1.calculateOutput(numpy.array([out_h1, out_h2]))

        return out_o1


def mse_loss(true, pred):
    return ((true - pred) ** 2).mean()


network = NeuralNetwork()
x = numpy.array([2, 3])
print(network.feedforward(x))
