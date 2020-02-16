import sys
import numpy as np
from keras.datasets import mnist
from MLUtils import MLUtils
from ActivationFunctions import ActivationFunctions as activFuncs
import math

'''
add convolutions
'''

np.random.seed(1)

#   ----    params  ----    #
train = True
test = True
load = True
visualize = True
dropout = True
regularization = False
alpha = 0.1
alphaDecay = 0.0
minAlpha = 0.001
iterations = 60000
batchSize = 32
hiddenLayerSize = 2048
inputSize = 28*28
numLabels = 10

weightScaler = 0.01
# weightScaler = 0.1

#   ----    create network  ----    #
layer1 = weightScaler * (2.0 * np.random.random((inputSize, hiddenLayerSize)) - 1.0)
layer2 = weightScaler * (2.0 * np.random.random((hiddenLayerSize, numLabels)) - 1.0)

if train or test:
    #   fetch data
    (rawTrainingData, rawTrainingLabels), (rawTestData, rawTestLabels) = mnist.load_data()

    #   ----    prep training  ----    #
    trainingData = rawTrainingData.reshape(60000, 28*28)
    trainingData = trainingData / 255.0
    trainingLabels = MLUtils.oneHot(rawTrainingLabels)

    #   ----    prep testing   ----    #
    testData = rawTestData.reshape(10000, 28*28)
    testData = testData / 255.0
    testLabels = MLUtils.oneHot(rawTestLabels)

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
        inputData = trainingData[batchStartIndex:batchEndIndex]
        labels = trainingLabels[batchStartIndex:batchEndIndex]

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

        alpha = max(alpha - alphaDecay, minAlpha)

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
    layers = [layer1, layer2]
    MLUtils.visualizeLayers(layers)

#   ----    test    ----    #
testIterations = len(testData)
if test:
    error = 0.0
    numCorrect = 0
    for i in range(testIterations):
        
        index = i
        inputData = testData[index:index+1]
        labels = testLabels[index:index+1]

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