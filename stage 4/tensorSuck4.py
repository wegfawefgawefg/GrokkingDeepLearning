import numpy as np
from Tensor import Tensor
from SGD import SGD
from Layer import ( Sequential,
                    Linear, 
                    MSELoss,
                    Tanh,
                    Sigmoid,
                    )

np.random.seed(0)

data   = Tensor( np.array([[0,0],[0,1],[1,0],[1,1]]),   autograd=True)
target = Tensor( np.array([[0],[1],[0],[1]]),           autograd=True)


model = Sequential( [Linear(2, 3), Tanh(), Linear(3, 1), Sigmoid()] )
optimizer = SGD(
    parameters = model.getParameters(), 
    alpha = 0.07)  
criterion = MSELoss()

for i in range(0, 10):
    pred = model.forward(data)
    loss = criterion.forward(pred, target)
    loss.backprop()
    optimizer.step()

    print(loss)