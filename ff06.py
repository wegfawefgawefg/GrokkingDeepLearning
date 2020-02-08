'''
learning single neuron
'''

input = 4.0
weight = 2.0
alpha = 0.01
target = 0.4

for i in range(0, 1000):
    output = weight * input
    error = output - target
    derivative = input * error
    delta = alpha * derivative
    weight = weight - delta
    print("output: " + str(output) + ", error: " + str(error))