# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:38:34 2019

@author: Victor
"""
import numpy as np
from adaptive.filtros.filtro_base import Filter

class NLMS(Filter):
    '''
    Implements the Normalized LMS algorithm for COMPLEX valued data.
    (Algorithm 4.3 - book: Adaptive Filtering: Algorithms and Practicalmplementation, Diniz)
    '''
    def __init__(self, n, mu=0.1, gamma=1e-6, w='zeros'):
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('A ordem do filtro deve ser um inteiro.')
        self.mu = mu
        self.gamma = gamma
        self.init_weights(w, self.n)
        self.w_history = False
    
    def train(self, d, x):
        if not len(d) == len(x):
            raise ValueError('Os vetores de entrada e valores desejados devem ter o mesmo tamanho.')
        #prefixed input
        x = np.insert(x, 0, np.zeros(self.n))
        y = np.zeros(len(d))
        e = np.zeros(len(d))
        #loop de treinamento
        for i in range(len(d)):
            regressor = x[i:(i+self.n)]
            y[i] = np.dot(self.w, regressor)
            e[i] = d[i] - y[i]
            self.w += (self.mu / (self.gamma + np.dot(regressor, regressor))) * e[i] * regressor
        return y, e