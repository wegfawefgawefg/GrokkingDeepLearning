import sys
import numpy as np
from scipy import signal
import math
import skimage.measure
import random
from collections import Counter

import MLUtils as MLUtils
import ActivationFunctions as activFuncs
import fillingBlanksData as fbd

'''
-finish chapter 11 god damn you are slow as fuck you spent 2 weeks on conv
'''

np.random.seed(1)

#   ----    params  ----    #
train = True
test = True
load = False
alpha = 0.05

iterations = 2
hiddenSize = 50
window = 2
negative = 5

amountOfData = 25000
numTestData = 1000
testDataStartIndex = amountOfData - numTestData
iterations = testDataStartIndex
numLabels = 1

#   ----    pre train   ----    #
#   fetch data
print("fetching data")
x = fbd.input_dataset
wordToIndex = fbd.word2index
concatenated = fbd.concatenated
vocab = fbd.vocab
print("data fetched")

#   ----    create network  ----    #
layer1 = (np.random.rand( len(vocab), hiddenSize ) - 0.5 ) * 0.2 
layer2 = 0.0 * np.random.rand(len(vocab), hiddenSize )

layer2Target = np.zeros(negative + 1)
layer2Target[0] = 1

def similar(target='beautiful'):
  target_index = wordToIndex[target]

  scores = Counter()
  for word,index in wordToIndex.items():
    raw_difference = layer1[index] - (layer1[target_index])
    squared_difference = raw_difference * raw_difference
    scores[word] = -math.sqrt(sum(squared_difference))
  return scores.most_common(10)

#   ----    train   ----    #
error = 0.0
numCorrect = 0
for rev_i, review in enumerate(x * iterations):
    for target_i in range(len(review)):
        #   get 5 random words from the entire review base
        randomSelectedIndecies = (np.random.rand(negative)*len(concatenated)).astype('int').tolist()
        selectedWords = list(concatenated[randomSelectedIndecies])
        targetSamples = [review[target_i]] + selectedWords

        leftContext = review[max(0,target_i-window):target_i]
        rightContext = review[target_i+1:min(len(review),target_i+window)]

        layer1Output = np.mean( layer1[ leftContext + rightContext ],axis=0)
        layer2Output = activFuncs.sigmoid( layer1Output.dot( layer2[targetSamples].T ))

        layer2Delta = layer2Output - layer2Target
        layer1Delta = layer2Delta.dot( layer2[targetSamples])

        layer1[leftContext+rightContext] -= layer1Delta * alpha
        layer2[targetSamples] -= np.outer(layer2Delta,layer1Output)*alpha
    if(rev_i % 250 == 0):
        sys.stdout.write(
            '\rProgress:'+str(
                rev_i/float(len(x)*iterations)*100.0)[0:4] + "      " + str(similar('terrible')))
    sys.stdout.write('\rProgress:'+str(rev_i/float(len(x)*iterations)*100.0)[0:4] ) 
print(similar('terrible'))

