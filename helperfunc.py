import numpy as np
from engine import *

#Random Functions
def randomint(start:int=0,end:int=None,shape:tuple=(1),req_grad:bool=False):
    if(type(end).__name__=='int'):
        return NVal(np.random.randint(start,end,size=shape),req_grad=req_grad)
    else :
        return NVal(np.random.randint(start,size=shape),req_grad=req_grad)

def randomn(shape,req_grad=False,xav=False):
    z=np.random.randn(*shape)
    if xav:
        z=z/np.sqrt(shape[0])
    return NVal(z,req_grad=req_grad)

def random(shape,req_grad=False):
    return NVal(np.random.random(shape),req_grad=req_grad)

def randomintlike(k:NVal,start:int,end:int=0,req_grad=False):
    return randomint(start,end,k.shape,req_grad=req_grad)

def randomnlike(k:NVal,req_grad=False,xav=False):
    return randomn(shape=k.shape,req_grad=req_grad,xav=xav)

#Zeros and Ones Functions
def zeros(shape,req_grad=False):
    return NVal(np.zeros(shape),req_grad=req_grad)

def ones(shape,req_grad=False):
    return NVal(np.ones(shape),req_grad=req_grad)

def zeroslike(k:NVal,req_grad=False):
    return zeros(shape=k.shape,req_grad=req_grad)

def oneslike(k:NVal,req_grad=False):
    return ones(shape=k.shape,req_grad=req_grad)

#Stat Funcs

def min(x:NVal,dim:int=-1,keepdims:bool=False):
    return x.min(dim=dim,keepdims=keepdims)

def argmin(x:NVal,dim:int=-1,keepdims:bool=False):
    return NVal(np.argminx(x._data,axis=dim,keepdims=keepdims))

def max(x:NVal,dim:int=-1,keepdims:bool=False):
    return x.max(dim=dim,keepdims=keepdims)

def argmax(x:NVal,dim:int=-1,keepdims:bool=False):
    return NVal(np.argmax(x._data,axis=dim,keepdims=keepdims))
