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

    def backprop(self,grad=None, grad_origin=None):
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

                if self.creationOp == "add":
                    for i in range(len(self.creators)):
                        self.creators[i].backprop(self.grad)

                elif self.creationOp == "sub":
                    for i in range(len(self.creators)):
                        self.creators[i].backprop(self.grad)

    ###############    TENSOR OPERATIONS   ##################
    def __add__(self, operand):
        if self.autograd and operand.autograd:
            return Tensor(
                data        = self.data + operand.data, 
                autograd    = True,
                creators    = [self, operand], 
                creationOp  = "add",
                )
        else:
            return Tensor( data = self.data + operand.data )
    
    def __sub__(self, operand):
        if self.autograd and operand.autograd:
            return Tensor(
                data        = self.data - operand.data, 
                autograd    = True,
                creators    = [self, operand], 
                creationOp  = "sub",
                )
        else:
            return Tensor( data = self.data - operand.data )

    ###############    REPR/STR   ##################
    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()