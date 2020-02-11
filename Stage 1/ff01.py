import numpy

inputs = [0, 1.0, 1.0]
weights = [0.0, 1.0, 1.0]

def dot(v1, v2):
    assert( len(v1) == len(v2))    
    pw_mult = [v2[i] * v1[i] for i in range(0, len(v1))]
    total = sum(pw_mult)
    return total

prod = dot(inputs, weights)
print(prod)