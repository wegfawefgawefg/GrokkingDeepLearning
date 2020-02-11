'''
many nueron single layer learning numpy
'''
import numpy as np
import sys
from datasets.mnist.MNIST_UTILS import MNIST_UTILS

rawInputs = MNIST_UTILS.getTrainingData()
inputs = rawInputs / 255.0
labels = MNIST_UTILS.getTrainingLabels()
targets = MNIST_UTILS.convertLabelsToTargets(labels)

layer1 = np.zeros((10,784)).T 

alpha = 0.0001
numTrainingCycles = 60000

def nn(inputs, weights):
    output = inputs.dot(weights)
    return output

sumErrors = []
sumError = 0.0
for i in range(0, numTrainingCycles):
    sys.stdout.write("\r{0:.2f}, er{1:.2f}".format(
        (float(i)/numTrainingCycles)*100, 
        sumError))
    sys.stdout.flush()

    index = 0
    pred = nn(inputs[index], layer1)
    errors = pred - targets[index]
    sumError = np.sum(np.square(errors))
    sumErrors.append(sumError)
    derivs = np.outer(errors, inputs[index])
    deltas = derivs * alpha
    layer1 = layer1 - deltas.T

# print()
# print("pred: " + str(pred))
# print("errors: " + str(errors))
# print("derivs: " + str(derivs))
# print("deltas: " + str(deltas))
# print("layer1: " + str(layer1))

print("---------------------------")

np.save("layer1.nn", layer1)
# layer1 = np.load("layer1.nn.npy")

introspects = []
for i in range(0, 10):
    introspect = layer1.T[i]
    introspect = introspect * 5000.0
    introspect = np.clip(introspect, 0.0, 255.0)
    introspects.append(introspect)

introspects = np.array(introspects)
introspectsImage = introspects.flatten()
MNIST_UTILS.showSomeImages(introspectsImage, 10)