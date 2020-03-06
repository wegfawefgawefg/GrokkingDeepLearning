import numpy as np
from collections import Counter
import sys
import random
import math
import re
import string 
from pprint import pprint

#########################################################

path = "F:\\_SSD_CODEING\\GrokkingDeepLearning\\stage 3\\babi\\tasksv11\\en\\qa1_single-supporting-fact_train.txt"
with open(path, 'r') as f:
    raw = f.readlines()

lines = []
for line in raw[:1000]:
    cleaned = line.lower().replace("\n", "").replace("\t", "")
    chars = [" " + char if not char.isalpha() else char for char in cleaned]
    chars = ''.join(chars)
    tokens = chars.split(' ')
    tokens = [token for token in tokens if not len(token) == 0]
    tokens = [token for token in tokens if token not in string.punctuation ]
    tokens = tokens[1:]
    if tokens[-1].isnumeric():
        tokens = tokens[:-1]
    lines.append( tokens )

print("Data sample: ")
pprint(lines[:3])
print("# " * 10)

#########################################################

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex/ex.sum(axis=0)

vocab = set()
for line in lines:
    for word in line:
        vocab.add(word)
vocab = list(vocab)
print("Vocab size" + str(len(vocab)) + '/n')
    
wordToIndex = {}
for i, word in enumerate(vocab):
    wordToIndex[word] = i

def wordsToIndices(words):
    indices = [wordToIndex[word] for word in words]
    return indices

#########################################################

np.random.seed(1)
embeddingSize = 10

embeddings = 0.1 * (np.random.rand(len(vocab), embeddingSize) - 0.5)

recurrent = np.eye(embeddingSize)

start = np.zeros(embeddingSize)

decoder =  0.1 * (np.random.rand(embeddingSize, len(vocab)) - 0.5)

oneHot = np.eye(len(vocab))

#########################################################

def predict(sent):
    layers = []
    layer = {}
    layer['hidden'] = start
    layers.append(layer)

    loss = 0

    # forward propagate
    for wordIndex in range(len(sent)):
        layer = {}

        # try to predict the next term
        layer['pred'] = softmax(layers[-1]['hidden'].dot(decoder))

        #   check to see what its prediction is for the CORRECT word
        #   #   why are we predicting the very first word???
        loss += -np.log( layer['pred'][sent[wordIndex]] )

        # generate the next hidden state
        layer['hidden'] = layers[-1]['hidden'].dot(recurrent) + embeddings[sent[wordIndex]]
        layers.append(layer)

    return layers, loss

#########################################################

for i in range(30000):
    alpha = 0.001
    lineNum = i % len(lines)
    line = lines[lineNum]
    sent = wordsToIndices(line)

    layers, loss = predict(sent)

    #   backprop
    for l in reversed(range(len(layers))):
        layer = layers[l]
        target = sent[l - 1]
        
        if(l > 0):  # if not the first layer
            layer['outputDelta'] = layer['pred'] - oneHot[target]
            hiddenDelta = layer['outputDelta'].dot(decoder.T)

            # if the last layer - don't pull from a later one because it doesn't exist
            if(l == len(layers) - 1):
                layer['hiddenDelta'] = hiddenDelta
            else:
                layer['hiddenDelta'] = hiddenDelta + layers[l+1]['hiddenDelta'].dot(recurrent.T)
        else: # if the first layer
            layer['hiddenDelta'] = layers[l+1]['hiddenDelta'].dot(recurrent.T)

    #   update weights
    sentLength = float(len(sent))
    start -= layers[0]['hiddenDelta'] * alpha / sentLength
    for l, layer in enumerate(layers[1:]):
        decoder -= np.outer(layers[l]['hidden'], layer['outputDelta']) * alpha / sentLength

        embedIndex = sent[l]
        embeddings[embedIndex] -= layers[l]['hiddenDelta'] * alpha / sentLength

        recurrent -= np.outer(layers[l]['hidden'], layer['hiddenDelta']) * alpha / sentLength
    
    if (i % 1000) == 0:
        print("perplexity:" + str(np.exp(loss/len(sent))))

sent_index = 4

l,_ = predict(wordsToIndices(lines[sent_index]))

print(lines[sent_index])

for i,each_layer in enumerate(l[1:-1]):
    input = lines[sent_index][i]
    true = lines[sent_index][i+1]
    pred = vocab[each_layer['pred'].argmax()]
    print("Prev Input:" + input + (' ' * (12 - len(input))) +\
          "True:" + true + (" " * (15 - len(true))) + "Pred:" + pred)