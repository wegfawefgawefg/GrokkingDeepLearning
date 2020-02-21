import sys
import numpy as np
from scipy import signal
import math
import skimage.measure

import MLUtils as MLUtils
import ActivationFunctions as activFuncs
import encodeIMDBData as edb

'''
-finish chapter 11 god damn you are slow as fuck you spent 2 weeks on conv
-fix alignment issue
'''

np.random.seed(1)

#   ----    params  ----    #
train = True
test = True
load = False
dropout = True
regularization = False
alphaDecay = False
visualize = False
alpha = 0.01
weightScaler = 0.01
amountOfData = 25000
numTestData = 1000
testDataStartIndex = amountOfData - numTestData
iterations = testDataStartIndex
hiddenSize = 128
numLabels = 1

if train or test:
    #   fetch data
    x, wordToIndex = edb.getIMDBData()
    y = edb.fetchIMDBLabels()

    #   ----    create network  ----    #
    layer1 = weightScaler * (2.0 * np.random.random(( len(wordToIndex), hiddenSize )) - 1.0) 
    layer2 = weightScaler * (2.0 * np.random.random(( hiddenSize, numLabels )) - 1.0)

#   ----    train   ----    #
if train:
    error = 0.0
    numCorrect = 0
    for i in range(iterations):
        if i%100 == 0:
           sys.stdout.write( "\r progress: %" + str((i/iterations)*100)[:5] ) 

        index = i        
        inputData = x[index]
        label = y[index]

        layer1Output = activFuncs.relu( np.sum(layer1[inputData], axis=0) )

        if dropout:
            dropoutMask = np.random.randint(2, size=layer1Output.shape)
            layer1Output *= dropoutMask
            layer1Output *= 2.0

        layer2Output = layer1Output.dot( layer2 )

        layer2Delta = label - layer2Output
        layer1Delta = layer2Delta.dot( layer2.T ) * activFuncs.derRelu(layer1Output)

        if dropout:
            layer1Delta *= dropoutMask

        layer2 += alpha * np.outer(layer1Output, layer2Delta)
        # layer2 += alpha * layer1Output.T.dot( layer2Delta )
        layer1[inputData] += alpha * layer1Delta

        #   stats
        error += np.sum(np.power(layer2Delta, 2))
        numCorrect += np.abs(layer2Delta) < 0.5
        
    trainingError = error / float(iterations) * 100.0
    trainingAccuracy = numCorrect / float(iterations) * 100.0
    sys.stdout.write(
        "\r iter: " + str(i) + \
        " training error: %"+str(trainingError)[0:5] + \
        " training accuracy: %"+str(trainingAccuracy)[0:5] + \
        " alpha: "+str(alpha)
    )
    print()

    np.save("saved_layers/layer1.nn", layer1)
    np.save("saved_layers/layer2.nn", layer2)

#   ----    visualize weights   ----    #
if load:
    layer1 = np.load("saved_layers/layer1.nn.npy")
    layer2 = np.load("saved_layers/layer2.nn.npy")

if visualize:
    # layers = [layer1, layer2]
    layers = [layer1, layer2]
    MLUtils.visualizeLayers(layers)

#   ----    test    ----    #
if test:
    error = 0.0
    numCorrect = 0
    for i in range(0, numTestData):
        index = testDataStartIndex + i
        inputData = x[index]
        label = y[index]

        layer1Output = activFuncs.lrelu( np.sum(layer1[inputData], axis=0) )
        layer2Output = layer1Output.dot( layer2 )

        layer2Delta = label - layer2Output
        
        #   stats
        error += np.sum(np.power(layer2Delta, 2))
        numCorrect += np.abs(layer2Delta) < 0.5
        
    testError = error / float(numTestData) * 100.0
    testAccuracy = numCorrect / float(numTestData) * 100.0
    sys.stdout.write(
        "\r iter: " + str(i) + \
        " test error: %"+str(testError)[0:5] + \
        " test accuracy: %"+str(testAccuracy)[0:5]
    )