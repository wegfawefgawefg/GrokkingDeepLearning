import numpy as np

def relu(x):
    return (x > 0) * x

def derRelu(x):
    return x > 0

def lrelu(x, leak=0.01):
    # return ((x > 0) * x) + (x <= 0) * leak
    return np.maximum(x, x * leak)

def derlrelu(x, leak=0.01):
    dx = np.ones_like(x)
    dx[x <= 0] = leak
    return dx

def tanh(x):
    return np.tanh(x)

def tanh2deriv(x):
    return 1 - (x ** 2)

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

def sigmoid(x):
    return 1/1 + np.exp(-x)