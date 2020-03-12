from pprint import pprint
import sys
import random
import math
from collections import Counter
import numpy as np
from Tensor import Tensor
from SGD import SGD
import pickle
from Layer import ( Sequential,
                    Linear, 
                    MSELoss,
                    Tanh,
                    Sigmoid,
                    Embedding,
                    CrossEntropyLoss,
                    RNNCell
                    )

np.random.seed(0)

###########     DATA WRANGLING      ##########

#   save the vocab
vocab = pickle.load( open( "vocab.p", "rb" ) )
#   save the indices
indices = pickle.load( open( "indices.p", "rb" ) )
#   save wordToIndex
wordToIndex = pickle.load( open( "wordToIndex.p", "rb" ) )

#############################################
data = indices

#   save embedd
embed = pickle.load( open( "embed.p", "rb" ) )
#   save the model
model = pickle.load( open( "model.p", "rb" ) )

criterion = CrossEntropyLoss()
optimizer = SGD(
    parameters = model.getParameters() + embed.getParameters(), 
    alpha = 0.05)  

#############################################
def generateSample(n=30, initChar = ' '):
    s = ""
    hidden = model.initHidden(batchSize = 1)
    input = Tensor(np.array([wordToIndex[initChar]]))
    for i in range(n):
        rnnInput = embed.forward(input)
        output, hidden = model.forward(input=rnnInput, hidden=hidden)
        output.data *= 10
        tempDist = output.softmax()
        tempDist /= tempDist.sum()

        m = (tempDist > np.random.rand()).argmax()
        c = vocab[m]
        input = Tensor(np.array([m]))
        s += c
    return s

#############################################

print(generateSample(n=200, initChar='\n'))

