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
test = False
load = False
dropout = True
regularization = False
alphaDecay = False
visualize = False
alpha = 4.0
weightScaler = 0.01
iterations = 60000
batchSize = 2
hiddenSize = 100
numLabels = 1

if train or test:
    #   fetch data
    x, corpus = edb.getIMDBData()
    y = edb.fetchIMDBLabels()

    #   ----    create network  ----    #
    layer1 = weightScaler * (2.0 * np.random.random(( len(corpus), hiddenSize )) - 1.0) 
    print("layer1shapep")
    print(layer1.shape)
    layer2 = weightScaler * (2.0 * np.random.random(( hiddenSize, numLabels )) - 1.0)

#   ----    train   ----    #
if train:
    error = 0.0
    numCorrect = 0
    batchRounds = int(iterations / batchSize)
    for i in range(batchRounds):
        if i%100 == 0:
           sys.stdout.write( "\r progress: %" + str((i/batchRounds)*100)[:5] ) 
        
        batchStartIndex = i * batchSize
        batchEndIndex = (i+1) * batchSize
        nonNpInputData = x[batchStartIndex:batchEndIndex]
        inputData = edb.oneHotSomeReviews(nonNpInputData, corpus)
        print("input data shape")
        print(inputData.shape)
        labels = y[batchStartIndex:batchEndIndex]

        layer1Output = activFuncs.lrelu( inputData.dot( layer1 ) )

        if dropout:
            dropoutMask = np.random.randint(2, size=layer1Output.shape)
            layer1Output *= dropoutMask
            layer1Output *= 2.0
        layer2Output = layer1Output.dot( layer2 )

        layer2Delta = (labels - layer2Output) / float(batchSize)
        layer1Delta = layer2Delta.dot( layer2.T ) * activFuncs.derlrelu(layer1Output)

        if dropout:
            layer1Delta *= dropoutMask

        layer2 += alpha * layer1Output.T.dot( layer2Delta )
        layer1 += alpha * inputData.T.dot( layer1Delta )

        #   stats
        error += np.sum(np.power(layer2Delta, 2))
        numCorrect += np.sum(np.argmax(layer2Output, axis=-1) == np.argmax(labels, axis=-1))
        
    trainingError = error / float(batchRounds) * 100.0
    trainingAccuracy = numCorrect / float(batchRounds * batchSize) * 100.0
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
    for i in range(testIterations):
        
        batchStartIndex = i * batchSize
        batchEndIndex = (i+1) * batchSize
        inputData = x[batchStartIndex:batchEndIndex]
        labels = y[batchStartIndex:batchEndIndex]

        layer1Output = activFuncs.lrelu( inputData.dot( layer1 ) )
        layer2Output = layer1Output.dot( layer2 )

        layer2Delta = labels - layer2Output

        #   stats
        error += np.sum(np.power(layer2Delta, 2))
        if int( np.argmax(layer2Output) ) == np.argmax(labels):
            numCorrect += 1
        
    testError = error / float(len(testData)) * 100.0
    testAccuracy = numCorrect / float(len(testData)) * 100.0
    sys.stdout.write(
        "\r iter: " + str(i) + \
        " test error: %"+str(testError)[0:5] + \
        " test accuracy: %"+str(testAccuracy)[0:5]
    )