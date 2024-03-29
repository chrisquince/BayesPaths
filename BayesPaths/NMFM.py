import re
import operator
import sys, getopt
import os
import pandas as p
import numpy as np
import random
import scipy.stats as ss
import scipy as sp
import scipy.misc as spm
import scipy.special as sps

import math
from operator import mul, truediv, eq, ne, add, ge, le, itemgetter

def elop(Xt, Yt, op):
    X = np.copy(Xt)
    Y = np.copy(Yt)
    try:
        X[X == 0] = np.finfo(X.dtype).eps
        Y[Y == 0] = np.finfo(Y.dtype).eps
    except ValueError:
        return op(X, Y)
    return op(X, Y)

class NMF():

    def __init__(self,V,M,rank,n_run=None,max_iter=None,min_change=None,prng=None):
        self.name = "NMF"
    
        if prng == None:
            self.prng = np.random.RandomState()
        else:
            self.prng = prng
            
        self.thresh = 1.0e-7
        self.V = V
        self.M = M
        self.rank = rank
        
        if n_run is None:
            self.n_run = 1
        else:
             self.n_run = n_run

        if max_iter is None:
            self.max_iter = 50000
        else:
             self.max_iter = max_iter

        if min_change is None:
            self.min_change = 1.0e-3
        else:
            self.min_change = min_change

    def factorize(self):
    
        bestdiv = sys.float_info.max
        bestW = None
        bestH = None
        for run in range(self.n_run):
            self.random_initialize()
        
            divl = 0.0
            div = self.div_objective()
            iter=0
            while iter < self.max_iter and math.fabs(divl - div) > self.min_change:
                self.div_update()
                self._adjustment()
                divl = div
                div = self.div_objective()
 
                iter += 1
            print(str(iter) + "," + str(div))
            if div < bestdiv:
                bestdiv = div
                bestW = self.W
                bestH = self.H
        div = bestdiv
        bestW = self.W
        bestH = self.H
        
        
    def factorizeH(self):
        divl = 0.0
        div = self.div_objective()
       
        for r in range(self.rank):
            Wr = self.W[:,r]
            medW = np.median(Wr) 
            Wr[Wr <= 0.1*medW] = np.finfo(self.W.dtype).eps
            Wr[Wr > 0.1*medW]  = 1.0
            
            self.W[:,r] = Wr
        
        iter=0
        while iter < self.max_iter and math.fabs(divl - div) > self.min_change:
            self.div_updateH()
            self._adjustment()
            divl = div
            div = self.div_objective()
 
            print(str(iter) + "," + str(div))

            iter += 1

    def factorizeW(self):
        iter=0
        div = self.div_objective()
        divl = 0.0
        while iter < self.max_iter and math.fabs(divl - div) > self.min_change:
            self.div_updateW()
            self._adjustment()
            divl = div
            div = self.div_objective()

            print(str(iter) + "," + str(div))

            iter += 1


    def random_initialize(self):
        self.max = (self.M*self.V).max()
            
        self.W = self.gen_dense(self.V.shape[0], self.rank)
        self.H = self.gen_dense(self.rank, self.V.shape[1])

        
    def gen_dense(self, dim1, dim2):
        return self.prng.uniform(0, self.max, (dim1, dim2))

    def _adjustment(self):
        """Adjust small values to factors to avoid numerical underflow."""
        self.H = np.maximum(self.H, np.finfo(self.H.dtype).eps)
        self.W = np.maximum(self.W, np.finfo(self.W.dtype).eps)



    def div_update(self):
        """Update basis and mixture matrix based on divergence multiplicative update rules."""
        H1 = np.dot(self.W.T,self.M)
        
        factorS = elop(self.V, np.dot(self.W, self.H), truediv)*self.M
        
        factorT = elop(np.dot(self.W.T, factorS), H1, truediv)
        
        self.H = np.multiply(self.H, factorT)
            
            
        W1 = np.dot(self.M,self.H.T)
        
        TW = elop(self.V, np.dot(self.W, self.H), truediv)*self.M
        
        factorW = elop(np.dot(TW, self.H.T), W1, truediv)
        
        self.W = np.multiply(self.W, factorW)  
        
    def div_updateHZ(self):
        """Update basis and mixture matrix based on divergence multiplicative update rules."""
        H1 = np.tile(np.asmatrix(self.Z.sum(0).T), (1, self.V.shape[1]))
        self.HZ = np.multiply(
            self.HZ, elop(np.dot(self.Z.T, elop(self.V*self.M, np.dot(self.Z, self.HZ), truediv)), H1, truediv))
 
    def fro_objective(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""
        R = (self.V - np.dot(self.W, self.H))*self.M
        return np.multiply(R, R).sum()
 
    def div_objective(self):
        """Compute divergence of target matrix from its NMF estimate."""
        Va = np.dot(self.W, self.H)
        return (np.multiply(self.V*self.M, np.log(elop(self.V, Va, truediv))) + self.M*(Va - self.V)).sum()

    def div_objectiveHZ(self):
        """Compute divergence of target matrix from its NMF estimate."""
        Va = np.dot(self.Z, self.HZ)
        return (np.multiply(self.V*self.M, np.log(elop(self.V, Va, truediv))) + self.M*(Va - self.V)).sum()

    def discretiseW(self):
        medW = np.median(self.W, axis=0)
        scaledW = self.W/medW
        scaledW[scaledW < 0.5] = 1.0e-5
        scaledW[scaledW >= 0.5] = 1.0 - 1.0e-5
        self.W = scaledW

    def div_updateW(self):
    
        W1 = np.dot(self.M,self.H.T)
        
        TW = elop(self.V, np.dot(self.W, self.H), truediv)*self.M
        
        factorW = elop(np.dot(TW, self.H.T), W1, truediv)
        
        self.W = np.multiply(self.W, factorW)  
        
 
    def div_updateH(self):
    
        H1 = np.dot(self.W.T,self.M)
        
        factorS = elop(self.V, np.dot(self.W, self.H), truediv)*self.M
        
        factorT = elop(np.dot(self.W.T, factorS), H1, truediv)
        
        self.H = np.multiply(self.H, factorT)
        
    
    def collapse(self):
        select = self.W.sum(axis=0) > 1.0e-3
        self.rank = select.sum()
        self.W = self.W[:,select]
        self.H = self.H[select,:]
    
    def __str__(self):
        return self.name
