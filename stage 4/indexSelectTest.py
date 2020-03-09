from Tensor import Tensor
import numpy as np

x = Tensor(np.eye(5), autograd=True)
pred = x.indexSelect(Tensor([[1,2,3],
                             [2,3,4]]))
pred.backprop()
print(x.grad)