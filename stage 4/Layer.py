import numpy as np
from Tensor import Tensor

class Layer:
    def __init__(self):
        self.parameters = []

    def getParameters(self):
        return self.parameters
    
class Linear(Layer):
    def __init__(self, numInputs, numOutputs):
        super().__init__()
        W = np.random.randn(numInputs, numOutputs) * np.sqrt(2.0/numInputs)
        self.weights = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(numOutputs), autograd=True)

        self.parameters.append(self.weights)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.mm(self.weights) + self.bias.expand(0, len(input.data))

class Sequential(Layer):
    def __init__(self, layers=[]):
        super().__init__()
        self.layers = layers

    def append(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def getParameters(self):
        parameters = []
        for layer in self.layers:
            parameters += layer.getParameters()
        return parameters

class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred - target) *(pred - target)).sum(0)