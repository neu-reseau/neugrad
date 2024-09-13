import numpy as np

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