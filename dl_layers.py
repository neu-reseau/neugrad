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
    
class Conv2d(NValModule):
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int,stride:int=1,padding:int=0):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.weight=NVal(np.random.randn(out_channels,in_channels,kernel_size,kernel_size)/np.sqrt(in_channels*kernel_size**2),req_grad=True)
        self.biad=NVal(np.zeros(out_channels),req_grad=True)
    def forprop(self,x):
        batch_size,in_channels,in_height,in_width=x.shape
        out_height=(in_height+2*self.padding-self.kernel_size)//self.stride+1
        out_width=(in_width+2*self.padding-self.kernel_size)//self.stride+1
        if self.padding>0:
            x=np.pad(x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))
        out=NVal(np.zeros((batch_size,self.out_channels,out_height,out_width)))
        for i in range(out_height):
            for j in range(out_width):
                h_start,h_end=i*self.stride,i*self.stride+self.kernel_size
                w_start,w_end=j*self.stride,j*self.stride+self.kernel_size
                rec_f=x[:,:,h_start:h_end,w_start:w_end]
                out[:,:,i,j]=(rec_f.reshape(batch_size,-1)@self.weight.reshape(self.out_channels,-1).T)+self.bias
        return out

class MaxPool2d(NValModule):
    def __init__(self,kernel_size:int,stride:int=None):
        super().__init__()
        self.kernel_size=kernel_size
        self.stride=stride if stride is not None else kernel_size
    def forprop(self,x):
        batch_size,in_channels,in_height,in_width=x.shape
        out_height=(in_height+2*self.padding-self.kernel_size)//self.stride+1
        out_width=(in_width+2*self.padding-self.kernel_size)//self.stride+1
        out=NVal(np.zeros((batch_size,self.out_channels,out_height,out_width)))
        for i in range(out_height):
            for j in range(out_width):
                h_start,h_end=i*self.stride,i*self.stride+self.kernel_size
                w_start,w_end=j*self.stride,j*self.stride+self.kernel_size
                pool_reg=x[:,:,h_start:h_end,w_start:w_end]
                out[:,:,i,j]=pool_reg.max(axis=(2,3))
        return out