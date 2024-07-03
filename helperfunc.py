import numpy as np
from engine import *

#Random Functions
def randomint(start,end,shape=(1),req_grad=False):
    return NVal(np.random.randint(start,end,size=shape))

def randomn(shape,req_grad=False,xav=False):
    z=np.random.randn(*shape)
    if xav:
        z=z/np.sqrt(shape[0])
    return NVal(z,req_grad=req_grad)

def random(shape,req_grad=False):
    return NVal(np.random.random(shape),req_grad=req_grad)

def randomintlike(k,start,end,req_grad=False):
    return randomint(start,end,k.shape,req_grad=req_grad)

def randomnlike(k,req_grad=False,xav=False):
    return randomn(shape=k.shape,req_grad=req_grad,xav=xav)

#Zeros and Ones Functions
def zeros(shape,req_grad=False):
    return NVal(np.zeros(shape),req_grad=req_grad)

def ones(shape,req_grad=False):
    return NVal(np.ones(shape),req_grad=req_grad)

def zeroslike(k,req_grad):
    return zeros(shape=k.shape,req_grad=req_grad)

def oneslike(k,req_grad):
    return ones(shape=k.shape,req_grad=req_grad)