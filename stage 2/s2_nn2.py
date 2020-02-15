import sys
import numpy as np
from keras.datasets import mnist
from MLUtils import MLUtils
import math

np.random.seed(10)

#   ----    params  ----    #
train = True
test = True
load = True
visualize = True
alpha = 0.001
iterations = 60000
hiddenLayerSize = 10
inputSize = 28*28
numLabels = 10

weightScaler = 0.01

#   ----    activation functions    ----    #
def relu(x):
    return (x > 0) * x
def derRelu(x):
    return x > 0

#   ----    create network  ----    #
layer1 = weightScaler * (2.0 * np.random.random((inputSize, hiddenLayerSize)) - 1.0)
layer2 = weightScaler * (2.0 * np.random.random((hiddenLayerSize, numLabels)) - 1.0)

#   ----    train   ----    #
if train:
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

    error = 0.0
    numCorrect = 0
    for i in range(iterations):
        
        index = i%len(trainingData)
        inputData = trainingData[index:index+1]
        labels = trainingLabels[index:index+1]

        layer1Output = relu( inputData.dot( layer1 ) )
        layer2Output = layer1Output.dot( layer2 )

        layer2Delta = labels - layer2Output
        layer1Delta = layer2Delta.dot( layer2.T ) * derRelu(layer1Output)

        layer2 += alpha * layer1Output.T.dot( layer2Delta )
        layer1 += alpha * inputData.T.dot( layer1Delta )

        #   stats
        error += np.sum(np.power(layer2Delta, 2))
        if int( np.argmax(layer2Output) ) == np.argmax(trainingLabels[index]):
            numCorrect += 1
        
    trainingError = error / float(len(trainingData))
    trainingAccuracy = numCorrect / float(len(trainingData))
    sys.stdout.write(
        "\r iter: " + str(i) + \
        " training error: "+str(trainingError)[0:5] + \
        " training accuracy: "+str(trainingAccuracy)[0:5]
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
        inputData = trainingData[index:index+1]
        labels = trainingLabels[index:index+1]

        layer1Output = relu( inputData.dot( layer1 ) )
        layer2Output = layer1Output.dot( layer2 )

        layer2Delta = labels - layer2Output

        #   stats
        error += np.sum(np.power(layer2Delta, 2))
        if int( np.argmax(layer2Output) ) == np.argmax(trainingLabels[index]):
            numCorrect += 1
        
    testError = error / float(len(testData))
    testAccuracy = numCorrect / float(len(testData))
    sys.stdout.write(
        "\r iter: " + str(i) + \
        " test error: "+str(testError)[0:5] + \
        " test accuracy: "+str(testAccuracy)[0:5]
    )