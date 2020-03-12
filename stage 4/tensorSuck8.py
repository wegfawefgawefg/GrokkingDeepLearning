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

#############################################
save = False

###########     DATA WRANGLING      ##########
filePath = "F:\\_SSD_CODEING\\GrokkingDeepLearning\\stage 4\\shakespear.txt"
with open(filePath, 'r') as f:
    raw = f.read()

vocab = list(set(raw))
wordToIndex = {}
for i, word in enumerate(vocab):
    wordToIndex[word] = i
indices = [wordToIndex[c] for c in raw]
indices = np.array(indices)

if save:
    #   save the vocab
    pickle.dump( vocab, open( "vocab.p", "wb" ) )
    #   save the indices
    pickle.dump( indices, open( "indices.p", "wb" ) )
    #   save wordToIndex
    pickle.dump( wordToIndex, open( "wordToIndex.p", "wb" ) )


#############################################
data = indices

embed = Embedding(vocabSize=len(vocab), dim=512)
model = RNNCell(
    numInputs=512, 
    numHidden=512,
    numOutput=len(vocab))

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
batchSize = 32
bptt = 16
numBatches = int(indices.shape[0] / batchSize)

trimmedIndices = indices[:numBatches * batchSize]
batchedIndices = trimmedIndices.reshape(batchSize, numBatches)
batchedIndices = batchedIndices.transpose()

#   target is always shifted ahead one index
inputBatchedIndices = batchedIndices[0:-1]
targetBatchedIndices = batchedIndices[1:]

numBptt = int((numBatches - 1) / bptt)
inputBatches = inputBatchedIndices[:numBptt * bptt]
inputBatches = inputBatches.reshape(numBptt, bptt, batchSize)
targetBatches = targetBatchedIndices[:numBptt * bptt]
targetBatches = targetBatches.reshape(numBptt, bptt, batchSize)

def train(iterations=100):
    for i in range(iterations):
        totalLoss = 0
        nLoss = 0
        
        hidden = model.initHidden(batchSize=batchSize)
        for batchIndex in range(len(inputBatches)):
            hidden = Tensor(hidden.data, autograd=True)
            loss = None
            losses = []
            for t in range(bptt):
                input = Tensor(inputBatches[batchIndex][t], autograd=True)
                rnnInput = embed.forward(input=input)
                output, hidden = model.forward(input=rnnInput, hidden=hidden)

                target = Tensor(targetBatches[batchIndex][t], autograd=True)
                batchLoss = criterion.forward(output, target)
                losses.append(batchLoss)
                if t == 0:
                    loss = batchLoss
                else:
                    loss = loss + batchLoss
                
            for loss in losses:
                ""
            loss.backprop()
            optimizer.step()
            totalLoss += loss.data
            log = "\r Iter:" + str(i)
            log += " - Batch " + str(batchIndex + 1) + "/" + str(len(inputBatches))
            log += " - Loss:" + str(np.exp(totalLoss / (batchIndex + 1)))
            if batchIndex == 0:
                log += " - " + generateSample( 70, '\n').replace("\n", " ")
            if (((batchIndex % 10) == 0) or (batchIndex - 1 == len(inputBatches))):
                sys.stdout.write(log)
        optimizer.alpha *= 0.99
        print()

train()
print(generateSample(n=200, initChar='\n'))

if save:
    #   save embedd
    pickle.dump( embed, open( "embed.p", "wb" ) )
    #   save the model
    pickle.dump( model, open( "model.p", "wb" ) )
