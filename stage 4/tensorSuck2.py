import numpy as np
from Tensor import Tensor
from SGD import SGD

np.random.seed(0)

data = Tensor(   np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True)
target = Tensor( np.array([[0],[1],[0],[1]]), autograd=True)

layers = []
layers.append( Tensor(np.random.rand(2, 3), autograd=True) )
layers.append( Tensor(np.random.rand(3, 1), autograd=True) )

optimizer = SGD(parameters=layers, alpha=0.1)

for i in range(0, 10):
    pred = data.mm(layers[0]).mm(layers[1])

    loss = ((pred - target)*(pred - target)).sum(0)
    
    loss.backprop(Tensor(np.ones_like(loss.data)))

    optimizer.step()

    print(loss)