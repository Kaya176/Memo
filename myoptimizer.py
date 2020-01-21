# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:34:23 2020

6장 코드 정리
-SGD
-Momentum
-AdaGrad
-
"""

import numpy as np

#모르겠으니 하나씩 살펴보자
class SGD:
    
    def __init__(self,lr=0.01):
        self.lr = lr
    
    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            
#진행방향에 가속도를 주면 어떨까? - Momentum
class Momentum:
    
    def __init__(self,lr = 0.01,momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self,params,grads):
        
        if self.v is None:
            self.v = dict()
            for key,val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            #전에 움직이던 방향을 추가해주는 방식으로 구현
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            #parameter을 업데이트해주는 부분
            params[key] += self.v[key]
            
#Momentum의 확장판 Nesterov accelerated gradient(NAG)
class NAG:
    pass

#모든 parameter마다 각기다른 learning rate를 부여하자! - AdaGrad
class AdaGrad:
    
    def __init__(self,lr = 0.01):
        self.lr = lr
        self.h = None
        
    def update(self,params,grads):
        
        if self.h is None:
            self.h = dict()
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            #분모는 이전 gradient를 제곱한 형태
            self.h[key] += grads[key] * grads[key]
            #parameter Update
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
#AdaGrad의 문제점 해결? - Adadelta            
class Adadelta:
    
    def __init__(self):
        self.e = None
        
    def update(self,params,grads):
        