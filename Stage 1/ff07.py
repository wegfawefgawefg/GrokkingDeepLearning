'''
single nueron single layer learning numpy
'''

import numpy as np

numWheels = np.array([0.0, 4, 6])
numDoors = np.array([0.5, 2, 3])
color = np.array([1.0, 2, 3])

target = 3.0
alpha = 0.01

inputs = np.array([numWheels[0], numDoors[0], color[0]])
layer1 = np.array([0.0, 1.0, 1.0])

def nn(inputs, weights):
    output = inputs.dot(weights)
    return output
    
for i in range(0, 1000):
    pred = nn(inputs, layer1)
    print("pred: " + str(pred))

    error = pred - target
    print("error: " + str(error))

    derivs = inputs * error
    print("derivs: " + str(derivs))

    deltas = derivs * alpha
    print("deltas: " + str(deltas))

    layer1 = layer1 - deltas
    print("layer1: " + str(layer1))
