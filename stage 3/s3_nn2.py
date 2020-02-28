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
-finish and move onto recurrent embeddings, you are slow as fuck you spent 2 weeks on convnets
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
#   layer 1 holds a bunch of high dimensional vectors: 1 vector for each word in the corpus
#   #   each number in the vector is some arbitrary parameter 
#   #   #   (would take serious investigation to determine what any individual parameter represents)
#   #   #   the meanings of the parameters is the same across word vectors... probably (hopefully) ((maybe))
layer1 = (np.random.rand( len(vocab), hiddenSize ) - 0.5 ) * 0.2
#   layer 2 is the word vectors probabilistic relationships with the target 
layer2 = 0.0 * np.random.rand(len(vocab), hiddenSize )

#   target is always 1 for this reviews selected word, and 0 for the random pool of words
layer2Target = np.zeros(negative + 1)
layer2Target[0] = 1

#   outputs the "similarity" of words.
#   #   just subtracts their word vectors, and then sorts by the smallest difference
#   #   #   works out to be semantically relevant
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
#   go through every word in every review
for rev_i, review in enumerate(x * iterations):
    for target_i in range(len(review)):
        #   get 5 random words from the entire corpus ("negative" sampling)
        randomSelectedIndecies = (np.random.rand(negative)*len(concatenated)).astype('int').tolist()
        selectedWords = list(concatenated[randomSelectedIndecies])
        #   put the random words in a pile with the target word from this review, 
        #   #  to generalize patterns from inside this review to words outisde outside this review
        targetSamples = [review[target_i]] + selectedWords

        #   grab words to left and right of target in review
        leftContext = review[max(0,target_i-window):target_i]
        rightContext = review[target_i+1:min(len(review),target_i+window)]

        #   fetch word vectors from layer 1, and aggregate
        layer1Output = np.mean( layer1[ leftContext + rightContext ],axis=0)
        #   compute probabilistic relationships between word vector mean and target word
        layer2Output = activFuncs.sigmoid( layer1Output.dot( layer2[targetSamples].T ))

        #   get error between computed probabilistic relatinships between word vector mean and real relationship between words
        layer2Delta = layer2Output - layer2Target
        #   multiply that error back through the probabability compute layer to fetch the individual word vector representation errors
        layer1Delta = layer2Delta.dot( layer2[targetSamples])

        #   adjust the word vectors by the error
        layer1[leftContext+rightContext] -= layer1Delta * alpha
        #   adjust aggregate probability computers by the error
        layer2[targetSamples] -= np.outer(layer2Delta,layer1Output)*alpha

    #   print out some similar word vectors during training
    #   #   gives you an idea of what relationships its learning
    if(rev_i % 250 == 0):
        sys.stdout.write(
            '\rProgress:'+str(
                rev_i/float(len(x)*iterations)*100.0)[0:4] + "      " + str(similar('terrible')))
    sys.stdout.write('\rProgress:'+str(rev_i/float(len(x)*iterations)*100.0)[0:4] ) 
print(similar('terrible'))

