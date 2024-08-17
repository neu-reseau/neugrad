import numpy as np
from engine import *
from helperfunc import *

class NValModule:
    def __init__(self):
        pass
    def __call__(self, x):
        return self.forprop(x)
    def parameters(self):
        pass
    def train(self):
        self.mode='train'
        for _,param in self.__dict__.items():
            if isinstance(param,NValModule):
                param.train()
    def eval(self):
        self.mode='eval'
        for _,param in self.__dict__.items():
            if isinstance(param,NValModule):
                param.eval()

class Linear_Regression(NValModule):
    def __init__(self,in_shape:int,out_shape:int,bias:bool=True):
        super().__init__()
        self.w=nval(np.random.randn(in_shape,out_shape)/np.sqrt(in_shape),req_grad=True)
        self.b=nval(np.zeros(out_shape),req_grad=True)
        self.req_bias=bias

    def forprop(self,x):
        k=x@self.w
        if self.req_bias:
            k+=self.b   
        return k 
    
class Dropout(NValModule):
    def __init__(self,drop_val):
        super().__init__()
        self.p=drop_val
        self.mode='train'
    def forprop(self,k):
        if self.mode=='eval':
            return k
        m=randomn(k.shape)>self.p
        data=k.masked_fill(m,0)
        data=data/(1-self.p)
        return data