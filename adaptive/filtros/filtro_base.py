# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:07:22 2019

@author: Victor
"""
import numpy as np

class Filter():
    '''
    Classe base para classes de filtros adaptativos.
    '''
    def init_weights(self, w, n=0):
        '''
        Inicializa os pesos do filtro adaptativo.
        '''
        if n == 0:
            n = self.n
        if type(w) == str:
            if w == 'random':
                w = np.random.normal(0, 0.5, n)
            elif w == 'zeros':
                w = np.zeros(n)
            else:
                raise ValueError('O argumento utilizado para inicialização de w não é válido.')
        elif len(w) == n:
            try:
                w = np.array(w, dtype = 'float64')
            except:
                raise ValueError('O argumento utilizado para inicialização de w não é válido.')
        else:
            raise ValueError ('O argumento utilizado para inicialização de w não é válido.')
        self.w = w
        
    def evaluate(self, x):
        return np.dot(self.w, x)
