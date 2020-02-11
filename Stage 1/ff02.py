import numpy

numWheels = [4, 4, 6]
numDoors = [4, 2, 3]
color = [1, 2, 3]

inputs = [numWheels[0], numDoors[0], color[0]]
weights = [0.0, 1.0, 1.0]

def dot(v1, v2):
    assert( len(v1) == len(v2))    
    pw_mult = [v2[i] * v1[i] for i in range(0, len(v1))]
    total = sum(pw_mult)
    return total

def nn(inputs, weights):
    output = dot(inputs, weights)
    return output
    
pred = nn(inputs, weights)

print(pred)