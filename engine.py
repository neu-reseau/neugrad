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

class NVal:
    def __init__(self,data,req_grad=False,op=None) -> None:
        self._data=np.array(data)
        self.req_grad=req_grad
        self.op=op
        self.children=[]
        self.shape=self._data.shape
        if self.req_grad:
            self.grad=np.zeros_like(data)

    def __repr__(self) -> str:
        return f"NVal({self._data})"
    
    def showval(self):
        return self._data
    
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
    
class Max:
    def forprop(self,a,dim,keepdims=False):
        data=np.max(a._data,axis=dim,keepdims=keepdims)
        req_grad=a.req_grad
        if keepdims:
            data=np.ones(a.shape)*data
        k=NVal(data,req_grad=req_grad,op=self)    
        self.parents(a,)
        a.children.append(k)
        self.cache=(a,data,dim)
        return k
    def backprop(self,dk,k):
        a,data,dim=self.cache
        if a.req_grad:
            max=data
            if a.shape!=dk.shape:
                dk=np.expand_dims(dk,axis=dim)
                dk=dk*np.ones_like(a._data)
                max=np.expand_dims(data,axis=dim)
                max=max*np.ones_like(a._data)
        da=dk*np.equal(a._data,max)
        a.backprop(da,k)

class Exp:
    def forprop(self,x):
        req_grad=x.req_grad
        data=np.exp(x._data)
        k=NVal(data,req_grad=req_grad,op=self)
        self.parents(x,)
        x.children.append(k)
        self.cache=x
        return k
    def backprop(self,dk,k):
        x=self.cache
        e=np.exp(x._data)
        if x.req_grad:
            dx=e*dk
            x.backprop(dx,k)
    
class Div():
    def forprop(self,p,q):
        req_grad=p.req_grad or q.req_grad
        data=p._data/q._data
        if q._data==0:
            raise Exception("You cannot divide by zero")
        k=NVal(data,req_grad=req_grad,op=self)
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
        req_grad=p.req_grad or q.req_grad
        data=p._data*q._data
        k=NVal(data,req_grad=req_grad,op=self)
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
        req_grad=p.req_grad
        data=p._data**q._data
        k=NVal(data,req_grad=req_grad,op=self)
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
        req_grad=p.req_grad
        data=-p._data
        k=NVal(data,req_grad=req_grad,op=self)
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
        req_grad=p.req_grad or q.req_grad
        data=p._data+q._data
        k=NVal(data,req_grad=req_grad,op=self)
        self.parents=(p,q)
        p.children.append(k)
        q.children.append(k)
        self.cache=p,q
        return k
    def backprop(self,dk,k):
        p,q=self.cache
        if p.req_grad:
            grad_dim=len(dk.shape)
            p_dim=len(p.shape)
            dp=0
            for _ in range(grad_dim-p_dim):
                dp=dp.sum(axis=0)
            for n,dim in enumerate(p.shape):
                if dim==1:
                    dp=dp.sum(axis=n,keepdims=True)
            p.backprop(dp,k)
        if q.req_grad:
            grad_dim=len(dk.shape)
            q_dim=len(q.shape)
            dq=0
            for _ in range(grad_dim-q_dim):
                dq=dq.sum(axis=0)
            for n,dim in enumerate(q.shape):
                if dim==1:
                    dq=dq.sum(axis=n,keepdims=True)
            q.backprop(dq,k)

