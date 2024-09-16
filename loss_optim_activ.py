import numpy as np
from dl_layers import *

#optim

class Adam:
    def __init__ (self,params,learning_rate=1e-3,beta=(0.9,0.99),ep=1e-8,regz=0):
        self.params=params
        self.learning_rate=learning_rate
        self.b1=beta[0]
        self.b2=beta[1]
        self.ep=ep
        self.regz=regz
        for p in params:
            p.m=np.zeros(params.shape)
            p.v=np.zeros(params.shape)

    def step(self):
        for p in self.params:
            p.m=(p.m*self.b1+1(1-self.b1)*p.grad)
            p.v=(p.v*self.b2+1(1-self.b2)*np.square(p.grad))
            p._data=p._data-(self.learning_rate*p.m)/(np.sqrt(p.v)+self.eps)-self.regz*self.learning_rate*p._data
    
    def grad_zero(self):
        for p in self.params():
            p.grad_zero()

class SGD:
    def __init__(self,params,learning_rate=10e-3,regz=0):
        self.params=params
        self.learning_rate=learning_rate
        self.regz=regz
    
    def step(self):
        for p in self.params:
            p._data=p._data-(self.learning_rate*p.grad)-(self.learning_rate*self.regz*p._data)

    def grad_zero(self):
        for p in self.params:
            p.grad_zero()

#Loss

class MSELoss(NValModule):
    def __init__(self):
        super().__init__()
    def forprop(self,y_pred,y_true):
        return np.mean((y_pred-y_true)**2)

class BinaryCrossEntropyLoss(NValModule):
    def __init__(self):
        super().__init__()
    def forprop(self,y_pred,y_true):
        ep=1e-15
        y_pred=np.clip(y_pred,ep,1-ep)
        return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
    
class HingeLoss(NValModule):
    def __init__(self):
        super().__init__()
    def forprop(self,y_pred,y_true):
        return np.mean(np.maximum(0,1-y_pred*y_true))
    
#Activation Funcs
    
class ReLU():
    def forprop(self, p):
        forp = p._data
        forp[forp < 0] = 0
        k=NVal(forp,req_grad=p.req_grad,op=self)
        self.parents=(p,)
        p.children.append(k)
        self.cache=p
        return k

    def backprop(self, dk, k):
        p=self.cache
        if p.req_grad:
            dp = (p._data > 0) * dk
            p.backprop(dp, k)


class Sigmoid():
    def forprop(self, p):
        forp = (1./ (1 + np.exp(-p._data)))
        k=NVal(forp,req_grad=p.req_grad,op=self)
        self.parents=(p,)
        p.children.append(k)
        self.cache = p
        self.out = forp
        return k

    def backprop(self, dk, k):
        p=self.cache
        if p.req_grad:
            dp = (self.out * (1. - self.out)) * dk
            p.backprop(dp, k)


class Tanh():
    def forprop(self, p):
        forp = (np.exp(2 * p._data) - 1) / (np.exp(2 * p._data) + 1)
        k=NVal(forp,req_grad=p.req_grad,op=self)
        self.parents=(p,)
        p.children.append(k)
        self.cache = p
        self.out = forp
        return k

    def backprop(self, dk, k):
        p=self.cache
        if p.req_grad:
            dp = (1. - (self.out ** 2)) * dk
            p.backprop(dp, k)