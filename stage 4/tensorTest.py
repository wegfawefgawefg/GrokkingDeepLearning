#   tensor test
import numpy as np
from Tensor import Tensor

a = Tensor([1,2,3,4,5], autograd=True)
b = Tensor([2,2,3,4,5], autograd=True)
f = Tensor([3,2,3,4,5], autograd=True)
g = Tensor([4,2,3,4,5], autograd=True)

c = a + b
d = f + c
e = g + c
k = d - e
k.backprop()

tensors = [a,b,c,d,e,f,g,k]
for tensor in tensors:
    print(tensor.grad)