import numpy as np

class Tensor:
    def __init__(self, 
                 data, 
                 autograd = False,
                 creators = None, 
                 creationOp = None,
                 id = None):
        if id is None:
            self.id = np.random.randint(0, 100000)
        else:
            self.id = id
        self.data = np.array(data)
        self.autograd = autograd
        self.grad = None
        self.creationOp = creationOp
        self.creators = creators

        self.children = {}
        if creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if(cnt != 0):
                return False
        return True    

    def backprop(self, grad=None, grad_origin=None):
        if self.autograd:
            #   so you dont have to pass in ones to backprop from end of network
            if(grad is None):
                grad = Tensor(np.ones_like(self.data))

            #   havent yet pooled all gradients from children yet
            if(grad_origin is not None):
                if(self.children[grad_origin.id] == 0):
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            # grads must not have grads of their own
            assert grad.autograd == False
            
            # only continue backpropping if there's something to
            # backprop into and if all gradients (from children)
            # are accounted for override waiting for children if
            # "backprop" was called on this variable directly
            if(self.creators is not None and 
               (self.all_children_grads_accounted_for() or 
                grad_origin is None)):

                if(self.creationOp == "add"):
                    self.creators[0].backprop(self.grad, self)
                    self.creators[1].backprop(self.grad, self)
                    
                if(self.creationOp == "sub"):
                    self.creators[0].backprop(Tensor(self.grad.data), self)
                    self.creators[1].backprop(Tensor(self.grad.__neg__().data), self)

                if(self.creationOp == "mul"):
                    new = self.grad * self.creators[1]
                    self.creators[0].backprop(new , self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backprop(new, self)                    
                    
                if(self.creationOp == "mm"):
                    c0 = self.creators[0]
                    c1 = self.creators[1]
                    new = self.grad.mm(c1.transpose())
                    c0.backprop(new)
                    new = self.grad.transpose().mm(c0).transpose()
                    c1.backprop(new)
                    
                if(self.creationOp == "transpose"):
                    self.creators[0].backprop(self.grad.transpose())

                if("sum" in self.creationOp):
                    dim = int(self.creationOp.split("_")[1])
                    self.creators[0].backprop(self.grad.expand(dim,
                                                               self.creators[0].data.shape[dim]))

                if("expand" in self.creationOp):
                    dim = int(self.creationOp.split("_")[1])
                    self.creators[0].backprop(self.grad.sum(dim))
                    
                if(self.creationOp == "neg"):
                    self.creators[0].backprop(self.grad.__neg__())

                if(self.creationOp == "sigmoid"):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backprop(self.grad * (self * (ones - self)))

                if(self.creationOp == "tanh"):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backprop(self.grad * (ones - (self * self)))

                if(self.creationOp == "indexSelect"):
                    newGrad = np.zeros_like(self.creators[0].data)
                    indices_ = self.indexSelectIndices.data.flatten()
                    grad_ = grad.data.reshape( len(indices_), -1 )
                    for i in range(len(indices_)):
                        newGrad[indices_[i]] += grad_[i]
                    self.creators[0].backprop(Tensor(newGrad))

                if(self.creationOp == "crossEntropy"):
                    dx = self.softmaxOutput - self.targetDist
                    self.creators[0].backprop(Tensor(dx))

    ###############    TENSOR OPERATIONS   ##################
    def __add__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                        autograd=True,
                        creators=[self,other],
                        creationOp="add")
        return Tensor(self.data + other.data)

    def __neg__(self):
        if(self.autograd):
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creationOp="neg")
        return Tensor(self.data * -1)
    
    def __sub__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self,other],
                          creationOp="sub")
        return Tensor(self.data - other.data)
    
    def __mul__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self,other],
                          creationOp="mul")
        return Tensor(self.data * other.data)    

    def sum(self, dim):
        if(self.autograd):
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creationOp="sum_"+str(dim))
        return Tensor(self.data.sum(dim))
    
    def expand(self, dim,copies):

        trans_cmd = list(range(0,len(self.data.shape)))
        trans_cmd.insert(dim,len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)
        
        if(self.autograd):
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creationOp="expand_"+str(dim))
        return Tensor(new_data)
    
    def transpose(self):
        if(self.autograd):
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creationOp="transpose")
        
        return Tensor(self.data.transpose())
    
    def mm(self, x):
        if(self.autograd):
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self,x],
                          creationOp="mm")
        return Tensor(self.data.dot(x.data))

    def sigmoid(self):
        if(self.autograd):
            return Tensor(
                data=1 / (1 + np.exp(-self.data)),
                autograd=True,
                creators=[self],
                creationOp="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))
    
    def tanh(self):
        if(self.autograd):
            return Tensor(
                data=np.tanh(self.data),
                autograd=True,
                creators=[self],
                creationOp="tanh")
        return Tensor(np.tanh(self.data))

    def indexSelect(self, indices):
        if(self.autograd):
            new = Tensor(
                self.data[indices.data],
                autograd=True,
                creators=[self],
                creationOp="indexSelect")
            new.indexSelectIndices = indices
            return new
        return Tensor(self.data[indices.dat])

    def softmax(self):
        temp = np.exp(self.data)
        softmaxOutput = temp / np.sum(  temp,
                                        axis=len(self.data.shape)-1,
                                        keepdims=True)
        return softmaxOutput

    def crossEntropy(self, targetIndices):
        temp = np.exp(self.data)
        softmaxOutput = temp / np.sum(temp,
                                    axis=len(self.data.shape)-1,
                                    keepdims=True)
        
        t = targetIndices.data.flatten()
        p = softmaxOutput.reshape(len(t),-1)
        targetDist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (targetDist)).sum(1).mean()
    
        if(self.autograd):
            out = Tensor(loss,
                        autograd=True,
                        creators=[self],
                        creationOp="crossEntropy")
            out.softmaxOutput = softmaxOutput
            out.targetDist = targetDist
            return out

        return Tensor(loss)
        

    ###############    REPR/STR   ##################
    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()