'''
deep learning relu
'''
import numpy as np
import sys
from datasets.mnist.MNIST_UTILS import MNIST_UTILS
import math
from statistics import mean

rawInputs = MNIST_UTILS.getTrainingData()
inputs = rawInputs / 255.0
labels = MNIST_UTILS.getTrainingLabels()
targets = MNIST_UTILS.convertLabelsToTargets(labels)

hiddenSize = int(math.pow(3, 2))
# hiddenSize = 10

layer1 = 0.0001 * (2.0 * np.random.random((784,hiddenSize)) - 1.0)
layer2 = 0.0001 * (2.0 * np.random.random((hiddenSize,10)) - 1.0)

alpha = 0.005
numTrainingCycles = 60000
# numTrainingCycles = 1000

def relu(x):
    return (x > 0.0) * x
def reluDer(y):
    return y > 0.0

train = False
if train:
    sumError = 0.0
    errors = [100.0] * 5
    for i in range(0, numTrainingCycles):
        roamingError = mean(errors[-5:])
        sys.stdout.write("\r{0:.2f}, er{1:.2f}".format(
            (float(i)/numTrainingCycles)*100, 
            roamingError))
        sys.stdout.flush()

        index = i
        layer1Out = relu(np.dot(inputs[index], layer1))
        layer2Out = np.dot(layer1Out, layer2)

        layer2Delta = layer2Out - targets[index]
        layer1Delta = layer2Delta.dot(layer2.T) * reluDer(layer1Out)
        # layer1Delta = (layer1Out - targets[index]) * reluDer(layer1Out)

        # layer1 -= (alpha * np.outer(inputs[index], layer1Delta))
        layer2 -= alpha * np.outer(layer1Out, layer2Delta)
        layer1 -= alpha * np.outer(inputs[index], layer1Delta)

        sumError = np.sum(np.square(layer2Delta))
        errors.append(sumError)
        
print("\n---------------------------")
printStats = False
if printStats:
    print(layer1Delta.shape)
    print("---------------------------")


# np.save("layer1.nn", layer1)
# np.save("layer2.nn", layer2)
layer1 = np.load("layer1.nn.npy")
layer2 = np.load("layer2.nn.npy")

layers = [layer1, layer2]
# layers = [layer1]
images = []
for l, layer in enumerate(layers):
    introspects = []
    for i in range(0, layer.shape[1]):
        introspect = layer.T[i]
        amp = 1.0
        if l == 0:
            amp = 2000.0
        elif l == 1:
            amp = 500.0
        introspect = introspect * amp
        introspects.append(introspect)
    introspects = np.array(introspects)
    introspectsImageGreyscale = introspects.flatten()
    introspectsImage = []
    for num in introspectsImageGreyscale:
        if num > 0.0:
            introspectsImage.append(0.0)
            introspectsImage.append(num)
            introspectsImage.append(0.0)
        elif num < 0.0:
            introspectsImage.append(-num)
            introspectsImage.append(0.0)
            introspectsImage.append(0.0)
        elif num == 0.0:
            introspectsImage.append(0.0)
            introspectsImage.append(0.0)
            introspectsImage.append(0.0)
    introspectsImage = np.asarray(introspectsImage)
    introspectsImage = np.clip(introspectsImage, 0.0, 255.0)
    image = MNIST_UTILS.genColorImageLine(
        introspectsImage, layer.shape[1],  int(math.sqrt(layer.shape[0])))
    images.append(image)
MNIST_UTILS.showColorImageLineGrid(images)