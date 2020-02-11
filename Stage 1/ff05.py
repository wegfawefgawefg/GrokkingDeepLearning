'''
multilayer numpy
'''

import numpy as np

numWheels = np.array([0.0, 4, 6])
numDoors = np.array([1.0, 2, 3])
color = np.array([1.0, 2, 3])

inputs = np.array([numWheels[0], numDoors[0], color[0]])
layer1 = np.array(
    [[0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0]]
).T

layer2 = np.array(
    [[0.0, 1.0],
    [1.0, 0.0],
    [0.0, 0.0]]
).T

weights = [layer1, layer2]

def nn(inputs, weights):
    output = inputs.dot(weights[0])
    output = output.dot(weights[1])
    return output
    
pred = nn(inputs, weights)

print(pred)