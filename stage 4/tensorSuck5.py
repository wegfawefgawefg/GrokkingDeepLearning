import numpy as np
from Tensor import Tensor
from SGD import SGD
from Layer import ( Sequential,
                    Linear, 
                    MSELoss,
                    Tanh,
                    Sigmoid,
                    Embedding,
                    )

np.random.seed(0)

data = Tensor( np.array([1,2,1,2]), autograd=True)
target = Tensor( np.array([[0],[1],[0],[1]]), autograd=True)

embed = Embedding(5,3)
model = Sequential( [embed, Tanh(), Linear(3, 1), Sigmoid()] )
criterion = MSELoss()

optimizer = SGD(
    parameters = model.getParameters(), 
    alpha = 0.1)  

for i in range(0, 10):
    pred = model.forward(data)
    loss = criterion.forward(pred, target)
    loss.backprop()
    optimizer.step()

    print(loss)