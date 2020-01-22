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
        return params
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
        
        return params
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
        
        return params
    
#AdaGrad의 문제점 해결? - Adadelta            
class Adadelta:
    
    def __init__(self):
        self.e = None
         
    def update(self,params,grads):
        pass
'''
참고 사이트 : http://aikorea.org/cs231n/neural-networks-3/#sgd
#1월21일

bias correction(*편향보정) 참고 사이트 : https://gaussian37.github.io/dl-dlai-bias_correction_exponentially_weighted_averages/
'''     

class RMSProp:
    
    def __init__(self,lr=0.01,decay_rate = 0.9):
        self.lr = lr
        self.h = None
        self.decay_rate = decay_rate
    
    def update(self,params,grads):
        
        if self.h is None :
            self.h = dict()
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            #AdaGrad 와 RMSProp의 차이점은 EMA를 사용하는가에 차이가 있다.
            self.h[key] = self.decay_rate*self.h[key] + (1-self.decay_rate)*grads[key]*grads[key]
            #아래 부분은 동일하다.
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key])+1e-7)
        
        return params
    
#Momentum + RMSProp = Adam?
class Adam:
    
    def __init__(self,lr = 0.01,momentum = 0.9 , decay_rate = 0.999):
        self.lr = lr
        self.momentum = momentum
        self.decay_rate = decay_rate
        self.h = None
        self.m = None
    
    #편향보정
    def BiasCorrection(self,parameter,rate,t):
        return parameter / (1-rate**t)
    
    def update(self,params,grads):
        
        #값 초기화
        if self.h is None:
            self.h = dict()
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)
        
        if self.m is None:
            self.m = dict()
            for key,val in params.items():
                self.m[key] = np.zeros_like(val)
        
        for t,key in enumerate(params.keys()):
            #gradient 부분은 Momentum의 아이디어를 적용
            self.m[key] = self.momentum * self.m[key] + (1-self.momentum) * grads[key]
            #RMSProp의 denominator부분을 적용
            self.h[key] = self.decay_rate * self.h[key] + (1-self.decay_rate) * grads[key]*grads[key]
            
            #Bias-Correction(편향보정) 해주기
            BC_m = self.BiasCorrection(self.m[key],self.momentum,t+1)
            BC_h = self.BiasCorrection(self.h[key],self.decay_rate,t+1)
            #Parameter Update
            params[key] -= self.lr*BC_m/(np.sqrt(BC_h) + 1e-8)
            
        return params