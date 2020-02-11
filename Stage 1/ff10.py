'''
two nueron two layer learning numpy
'''

import numpy as np
import sys

numWheels = np.array([0.5, 0.0, 4, 6])
numDoors = np.array([0.5, 0.5, 2, 3])
color = np.array([0.5, 1.0, 2, 3])

inputs = np.array([numWheels[0], numDoors[0], color[0]])
layer1 = np.array(
    [[0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],]
).T

layer2 = np.array(
    [[0.0, 1.0],
    [0.5, 0.5],]
).T

weights = [layer1, layer2]

targets = [1.0] * layer1.shape[1] 
alpha = 0.01
numTrainingCycles = 10000

def nn(inputs, weights):
    outputs = []

    output = inputs.dot(weights[0])
    output.append(output)
    
    output = inputs.dot(weights[1])
    outputs.append(output)

    return outputs
    
for i in range(0, numTrainingCycles):
    sys.stdout.write("\r{0}".format((float(i)/numTrainingCycles)*100))
    sys.stdout.flush()

    pred = nn(inputs, layer1)
    errors = pred - targets
    derivs = np.outer(errors, inputs)
    deltas = derivs * alpha
    layer1 = layer1 - deltas.T

print("pred: " + str(pred))
print("errors: " + str(errors))
print("derivs: " + str(derivs))
print("deltas: " + str(deltas))
print("layer1: " + str(layer1))