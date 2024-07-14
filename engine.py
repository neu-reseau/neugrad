import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List

#helper funcs
def list(data):
    if isinstance(data,List):
        return data
    else:
        return data.tolist()
def array(data):
    if isinstance(data,np.ndarray):
        return data
    elif isinstance(data,NVal):
        return data.toarray()
    else:
        return np.array(data)
def nval(data):
    if isinstance(data,NVal):
        return data
    else:
        return NVal(data)
    
#main tensor class (NVal)

class NVal:
    def __init__(self,data,req_grad=False,op=None) -> None:
        self._data=array(data)
        self.req_grad=req_grad
        self.op=op
        self.shape=self._data.shape
        self.children=[]
        if self.req_grad:
            self.grad=np.zeros_like(data,dtype=np.float64)

    def __repr__(self) -> str:
        return f"""NVal({self._data})"""
    
    def grad_zero(self):
        self.grad=np.zeros_like(self._data)

    def backprop(self,grad=None,k=None):
        if not self.req_grad:
            return "This tensor has req_grad set to False"
        if grad is None:
            grad=np.ones_like(self._data)
        self.grad+=grad
        if k is not None:
            self.children.remove(k)
        if self.op:
            if not self.children:
                self.op.backprop(self.grad,self)
    
    def showval(self):
        return self._data

    def __neg__(self):
        out=Neg()
        return out.forprop(self)
    
    def __add__(self,other):
        out=Add()
        return out.forprop(self,nval(other))
    
    def __sub__(self,other):
        out=Add()
        return out.forprop(self,-other)
    
    def __pow__(self,other):
        out=Pow()
        return out.forprop(self,nval(other))
    
    def __mul__(self,other):
        out=Mul()
        return out.forprop(self,nval(other))
    
    def __truediv__(self,other):
        out=Div()
        return out.forprop(self,nval(other))
    
    def max(self,dim=-1,keepdims=False):
        out=Max()
        return out.forprop(self,dim,keepdims=keepdims)
    
    def __matmul__(self,other):
        out=MatMul()
        return out.forprop(self,nval(other))
    
    def __gt__(self,other):
        return self._data>array(other)
    
    def __lt__(self,other):
        return self._data<array(other)
    
    def mean(self,dim=-1,keepdims=False):
        op=Mean()
        return op.forprop(self,dim,keepdims=keepdims)
    
    def transpose(self,*dims):
        op=Transpose()
        return op.forprop(self, *dims)

#Operation Classes
class Transpose:
    def forprop(self,p,*dims):
        forp=p._data.swapaxes(*dims)
        k=NVal(forp,req_grad=p.req_grad,op=self)
        self.parents=(p,)
        p.children.append(k)
        self.cache=(p,dims)
        return k
    def backprop(self,dk,k):
        p,dims=self.cache
        if p.req_grad:
            dp=dk.swapaxes(dims)
            p.backprop(dp,k)

class Mean:
    def forprop(self,p,dim,keepdims):
        forp=p._data.mean(axis=dim,keepdims=keepdims)
        k=NVal(forp,req_grad=p.req_grad,op=self)
        self.parents=(p,)
        p.children.append(k)
        self.cache=(p,dim)
        return k
    def backprop(self,dk,k):
        p,dim=self.cache
        if p.req_grad:
            dp=np.ones(p.shape)*dk
            dp=dp/(np.prod(np.array(p.shape)[dim]))
            p.backprop(dp,k)
    
class MatMul:
    def forprop(self,p,q):
        forp=p._data@q._data
        k=NVal(forp,req_grad=(p.req_grad or q.req_grad),op=self)
        self.parents=(p,q)
        p.children.append(k)
        q.children.append(k)
        self.cache=(p,q)
        return k
    def backprop(self,dk,k):
        p,q=self.cache
        if p.req_grad:
            dp=dk@q._data.swapaxes(-1,-2)
            grad_dim=len(dp.shape)
            p_dim=len(p.shape)
            for _ in range(grad_dim-p_dim):
                dp=dp.sum(axis=0)
            p.backprop(dp,k)
        if q.req_grad:
            dq=p._data.swapaxes(-1,-2)@dk
            #dq=dk@kp._data.swapaxes(-1,-2)
            grad_dim=len(dq.shape)
            q_dim=len(q.shape)
            for _ in range(grad_dim-q_dim):
                dq=dq.sum(axis=0)
            q.backprop(dq,k)
    
class Max:
    def forprop(self,p,dim,keepdims=False):
        forp=np.max(p._data,axis=dim,keepdims=keepdims)
        if keepdims:
            forp=np.ones(p.shape)*forp
        k=NVal(forp,req_grad=p.req_grad,op=self)    
        self.parents(p,)
        p.children.append(k)
        self.cache=(p,forp,dim)
        return k
    def backprop(self,dk,k):
        p,forp,dim=self.cache
        if p.req_grad:
            max=forp
            if p.shape!=dk.shape:
                dk=np.expand_dims(dk,axis=dim)
                dk=dk*np.ones_like(p._data)
                max=np.expand_dims(forp,axis=dim)
                max=max*np.ones_like(p._data)
        dp=dk*np.equal(p._data,max)
        p.backprop(dp,k)
    
class Div:
    def forprop(self,p,q):
        forp=p._data/q._data
        if q._data==0:
            raise Exception("You cannot divide by zero")
        k=NVal(forp,req_grad=(p.req_grad or q.req_grad),op=self)
        self.parents=(p,q)
        p.children.append(k)
        q.children.append(k)
        self.cache=(p,q)
        return k

    def backprop(self,dk,k):
        p,q=self.cache
        if p.req_grad:
            dp=dk*(1/q._data)
            grad_dim=len(dk.shape)
            p_dim=len(p.shape)
            for _ in range(grad_dim-p_dim):
                dp=dp.sum(axis=0)
            for n,dim in enumerate(p.shape):
                if dim==1:
                    dp=dp.sum(axis=n,keepdims=True)    
            p.backprop(dp,k)
        if q.req_grad:
            dq=-dk*p._data/(q._data**2)
            grad_dim=len(dk.shape)
            q_dim=len(q.shape)
            for _ in range(grad_dim-q_dim):
                dq=dq.sum(axis=0)
            for n,dim in enumerate(q.shape):
                if dim==1:
                    dq=dq.sum(axis=n,keepdims=True)    
            q.backprop(dq,k)


class Mul:
    def forprop(self,p,q):
        forp=p._data*q._data
        k=NVal(forp,req_grad=(p.req_grad or q.req_grad),op=self)
        self.parents=(p,q)
        p.children.append(k)
        q.children.append(k)
        self.cache=(p,q)
        return k
    def backprop(self,dk,k):
        p,q=self.cache
        if p.req_grad:
            dp=dk*q._data
            grad_dim=len(dk.shape)
            p_dim=len(p.shape)
            for _ in range(grad_dim-p_dim):
                dp=dp.sum(axis=0)
            for n,dim in enumerate(p.shape):
                if dim==1:
                    dp=dp.sum(axis=n,keepdims=True)    
            p.backprop(dp,k)
        if q.req_grad:
            dq=dk*p._data
            grad_dim=len(dk.shape)
            q_dim=len(q.shape)
            for _ in range(grad_dim-q_dim):
                dq=dq.sum(axis=0)
            for n,dim in enumerate(q.shape):
                if dim==1:
                    dq=dq.sum(axis=n,keepdims=True)    
            q.backprop(dq,k)

class Pow:
    def forprop(self,p,q):
        forp=p._data**q._data
        k=NVal(forp,req_grad=p.req_grad,op=self)
        p.children.append(k)
        self.cache=(p,q)
        return k
    def backprop(self,dk,k):
        p,q=self.cache
        if p.req_grad:
            dp=dk*(q._data*p._data**(q._data-1))
            grad_dim=len(dk.shape)
            p_dim=len(p.shape)
            for _  in range(grad_dim-p_dim):
                dp=dp.sum(axis=0)
            for n,dim in enumerate(p.shape):
                if dim==1:
                    dp=dp.su(axis=n,keepdims=True)
            p.backprop(dp,k)

    
class Neg:
    def forprop(self,p):
        forp=-p._data
        k=NVal(forp,req_grad=p.req_grad,op=self)
        self.parents=(p,)
        p.children.append(k)
        self.cache=p
        return k
    def backprop(self,dk,k):
        p=self.cache
        if p.req_grad:
            dp=-dk
            p.backprop(dp,k)    


class Add:
    def forprop(self,p,q):
        forp=p._data+q._data
        k=NVal(forp,req_grad=(p.req_grad or q.req_grad),op=self)
        self.parents=(p,q)
        p.children.append(k)
        q.children.append(k)
        self.cache=(p,q)
        return k
    def backprop(self,dk,k):
        p,q=self.cache
        if p.req_grad:
            dp=dk
            grad_dim=len(dk.shape)
            p_dim=len(p.shape)
            for _ in range(grad_dim-p_dim):
                dp=dp.sum(axis=0)
            for n,dim in enumerate(p.shape):
                if dim==1:
                    dp=dp.sum(axis=n,keepdims=True)
            p.backprop(dp,k)
        if q.req_grad:
            dq=dk
            grad_dim=len(dk.shape)
            q_dim=len(q.shape)
            for _ in range(grad_dim-q_dim):
                dq=dq.sum(axis=0)
            for n,dim in enumerate(q.shape):
                if dim==1:
                    dq=dq.sum(axis=n,keepdims=True)
            q.backprop(dq,k)

