from pprint import pprint
import sys
import random
import math
from collections import Counter
import numpy as np
from Tensor import Tensor
from SGD import SGD
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
filePath = "F:\\_SSD_CODEING\\GrokkingDeepLearning\\stage 3\\babi\\tasksv11\\en\\qa1_single-supporting-fact_train.txt"
with open(filePath, 'r') as f:
    rawLines = f.readlines()

lines = []
for line in rawLines:
    cleaned = line.lower().replace("\n","").replace("\t", " ").replace("?", " ?").replace(".", " .")
    lines.append(cleaned.split(" ")[1:])

#   pad that shit because numpy doesnt like rows of dif lengths
newLines = []
for line in lines:
    newLines.append(['-'] * (7 - len(line)) + line)
lines = newLines

vocab = set()
for line in lines:
    for word in line:
        vocab.add(word)
vocab = list(vocab)

wordToIndex = {}
for i, word in enumerate(vocab):
    wordToIndex[word] = i

def wordsToIndices(words):
    return [wordToIndex[word] for word in words if word in wordToIndex]

#   convert lines to indices arrays
linesAsIndices = []
for line in lines:
    linesAsIndices.append(wordsToIndices(line))

data = np.array(linesAsIndices)

embed = Embedding(vocabSize=len(vocab), dim=16)
model = RNNCell(
    numInputs=16, 
    numHidden=16,
    numOutput=len(vocab))

criterion = CrossEntropyLoss()
optimizer = SGD(
    parameters = model.getParameters() + embed.getParameters(), 
    alpha = 0.05)  

batchSize = 100
for i in range(1000):
    totalLoss = 0

    hidden = model.initHidden(batchSize=batchSize)

    for t in range(5):
        input = Tensor(data[0:batchSize, t], autograd=True)
        rnnInput = embed.forward(input=input)
        output, hidden = model.forward(input=rnnInput, hidden=hidden)
    
    target = Tensor(data[0:batchSize, t+1], autograd=True)
    loss = criterion.forward(output, target)
    loss.backprop()
    optimizer.step()
    totalLoss += loss.data
    if i % 200 == 0:
        printCorrect = (target.data == np.argmax(output.data, axis=1)).mean()
        printLoss = totalLoss / (len(data) / batchSize)
        print("loss: ", printLoss, "% correct: ", printCorrect)


batchSize = 1
hidden = model.initHidden(batchSize=batchSize)
for t in range(0,5):
    input = Tensor(data[0:batchSize, t], autograd=True)
    rnnInput = embed.forward(input=input)
    output, hidden = model.forward(input=rnnInput, hidden=hidden)

target = Tensor(data[0:batchSize, t+1], autograd=True)
loss = criterion.forward(output, target)

ctx = ""
for idx in data[0:batchSize][0][0:-1]:
    ctx += vocab[idx] + " "
print("Context:",ctx)
print("True:",vocab[target.data[0]])
print("Pred:", vocab[output.data.argmax()])