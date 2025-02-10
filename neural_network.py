import numpy

class NeuralNetwork:
    def __init__(self):
        weight = numpy.array([0,1])
        bias = 0

        self.h1 = Node(weight, bias)
        self.h2 = Node(weight, bias)
        self.o1 = Node(weight, bias)

        def feedforward(self, x):
            out_h1 = self.h1.calculateOutput(x)
            out_h2 = self.h2.calculateOutput(x)
            
            out_o1 = self.o1.calculateOutput(numpy.array([out_h1, out_h2]))

            return out_o1

network = NeuralNetwork()
x = numpy.array([2,3])
print (network.feedforward(x))
