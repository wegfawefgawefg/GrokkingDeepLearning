class SGD:
    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        for parameter in self.parameters:
            parameter.grad.data *= 0

    def step(self, zero=True):
        for parameter in self.parameters:
            parameter.data -= parameter.grad.data * self.alpha
            
            if zero:
                parameter.grad.data *= 0

