import numpy as np
from Tensor import Tensor

class Layer:
    def __init__(self):
        self.parameters = []

    def getParameters(self):
        return self.parameters

###############    SINGLE LAYERS   ##################
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

class Embedding(Layer):
    def __init__(self, vocabSize, dim):
        super().__init__()
        self.vocabSize = vocabSize
        self.dim = dim
        
        weights = (np.random.rand(vocabSize, dim) - 0.5) / dim
        self.weights = Tensor(weights, autograd=True)

        self.parameters.append(self.weights)

    def forward(self, input):
        return self.weights.indexSelect(input)

class RNNCell(Layer):
    def __init__(self, numInputs, numHidden, numOutput, activation='sigmoid'):
        super().__init__()
        self.numInputs = numInputs
        self.numHidden = numHidden
        self.numOutput = numOutput

        if(activation == 'sigmoid'):
            self.activation = Sigmoid()
        elif(activation == 'tanh'):
            self.activation = Tanh()
        else:
            raise Exception("no non-linearity... no activation function")

        self.weightsInput = Linear(numInputs, numHidden)
        self.weightsHidden = Linear(numHidden, numHidden)
        self.weightsOutput = Linear(numHidden, numOutput)

        self.parameters += self.weightsInput.getParameters()
        self.parameters += self.weightsHidden.getParameters()
        self.parameters += self.weightsOutput.getParameters()

    def forward(self, input, hidden):
        hiddenOut = self.weightsHidden.forward(hidden)
        combo = self.weightsInput.forward(input) + hiddenOut
        newHidden = self.activation.forward(combo)
        output = self.weightsOutput.forward(newHidden)
        return output, newHidden

    def initHidden(self, batchSize=1):
        return Tensor(np.zeros((batchSize, self.numHidden)), autograd=True)

###############    MODEL LAYERS   ##################
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

###############    ACTIVATION LAYERS   ##################
class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred - target) *(pred - target)).sum(0)

class CrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return pred.crossEntropy(target)

class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()

