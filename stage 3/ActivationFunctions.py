import numpy as np

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return (x > 0) * x
    
    @staticmethod
    def derRelu(x):
        return x > 0

    @staticmethod
    def lrelu(x, leak=0.01):
        # return ((x > 0) * x) + (x <= 0) * leak
        return np.maximum(x, x * leak)

    @staticmethod
    def derlrelu(x, leak=0.01):
        dx = np.ones_like(x)
        dx[x <= 0] = leak
        return dx