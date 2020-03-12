import numpy as np
from Tensor import Tensor

class Layer:
    def __init__(self):
        self.parameters = []

    def getParameters(self):
        return self.parameters

###############    SINGLE LAYERS   ##################
class Linear(Layer):
    def __init__(self, numInputs, numOutput, bias=True):
        super().__init__()
        self.useBias = bias
        W = np.random.randn(numInputs, numOutput) * np.sqrt(2.0/numInputs)
        self.weights = Tensor(W, autograd=True)
        self.parameters.append(self.weights)
        if self.useBias:
            self.bias = Tensor(np.zeros(numOutput), autograd=True)
            self.parameters.append(self.bias)

    def forward(self, input):
        if self.useBias:
            return input.mm(self.weights) + self.bias.expand(0, len(input.data))
        return input.mm(self.weights)

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

class LSTMCell(Layer):
    def __init__(self, numInputs, numHidden, numOutput):
        super().__init__()

        self.numInputs = numInputs
        self.numHidden = numHidden
        self.numOutput = numOutput

        self.xf = Linear(numInputs, numHidden)
        self.xi = Linear(numInputs, numHidden)
        self.xo = Linear(numInputs, numHidden)        
        self.xc = Linear(numInputs, numHidden)        
        
        self.hf = Linear(numHidden, numHidden, bias=False)
        self.hi = Linear(numHidden, numHidden, bias=False)
        self.ho = Linear(numHidden, numHidden, bias=False)
        self.hc = Linear(numHidden, numHidden, bias=False)        
        
        self.w_ho = Linear(numHidden, numOutput, bias=False)
        
        self.parameters += self.xf.getParameters()
        self.parameters += self.xi.getParameters()
        self.parameters += self.xo.getParameters()
        self.parameters += self.xc.getParameters()

        self.parameters += self.hf.getParameters()
        self.parameters += self.hi.getParameters()        
        self.parameters += self.ho.getParameters()        
        self.parameters += self.hc.getParameters()                
        
        self.parameters += self.w_ho.getParameters()        
    
    def forward(self, input, hidden):
        
        prevHidden = hidden[0]        
        prevCell = hidden[1]
        
        f = (self.xf.forward(input) + self.hf.forward(prevHidden)).sigmoid()
        i = (self.xi.forward(input) + self.hi.forward(prevHidden)).sigmoid()
        o = (self.xo.forward(input) + self.ho.forward(prevHidden)).sigmoid()        
        g = (self.xc.forward(input) + self.hc.forward(prevHidden)).tanh()        
        c = (f * prevCell) + (i * g)

        h = o * c.tanh()
        
        output = self.w_ho.forward(h)
        return output, (h, c)
    
    def initHidden(self, batchSize=1):
        h = Tensor(np.zeros((batchSize,self.numHidden)), autograd=True)
        c = Tensor(np.zeros((batchSize,self.numHidden)), autograd=True)
        h.data[:,0] += 1
        c.data[:,0] += 1
        return (h, c)

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

