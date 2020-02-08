'''
multilayer
'''

import numpy as np


numWheels = np.array([0.0, 4, 6])
numDoors = np.array([1.0, 2, 3])
color = np.array([1.0, 2, 3])

inputs = np.array([numWheels[0], numDoors[0], color[0]])
layer1 = np.array(
    [[0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0]]
)

layer2 = np.array(
    [[0.0, 1.0],
    [1.0, 0.0],
    [0.0, 0.0]]
)

weights = [layer1, layer2]

def dot(v1, v2):
    assert( len(v1) == len(v2))    
    pw_mult = [v2[i] * v1[i] for i in range(0, len(v1))]
    total = sum(pw_mult)
    return total

def vm_mult(vec, mat):
    assert(len(vec) == len(mat[0]))

    result = [0] * len(mat)
    for i, row in enumerate(mat):
        result[i] = dot(vec, row)
    return result

def nn(inputs, weights):
    output = vm_mult(inputs, weights[0])
    output = vm_mult(output, weights[1])
    return output
    
pred = nn(inputs, weights)

print(pred)