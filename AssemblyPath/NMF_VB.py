import uuid
import re
import operator
import sys, getopt
import os
import pandas as p
import numpy as np
import numpy.ma as ma
import random
import scipy.stats as ss
import scipy as sp
import scipy.misc as spm
import scipy.special as sps
from scipy.special import psi as digamma
from scipy.stats import truncnorm
from scipy.special import erfc
from scipy.special import erf

from copy import deepcopy
from copy import copy

import math
from math import floor
import subprocess
from subprocess import Popen, PIPE, STDOUT
from operator import mul, truediv, eq, ne, add, ge, le, itemgetter
import networkx as nx
import argparse

import itertools, math, scipy, time

import collections
from collections import deque
from collections import defaultdict
from collections import Counter
from numpy.random import RandomState

from Utils.UtilsFunctions import convertNodeToName
from Utils.UtilsFunctions import elop
from Utils.UtilsFunctions import expNormLogProb
from Utils.UtilsFunctions import TN_vector_expectation
from Utils.UtilsFunctions import TN_vector_variance
from Utils.UtilsFunctions import readRefAssign
from Utils.UnitigGraph import UnitigGraph
from BNMF_ARD.exponential import exponential_draw

from Utils.AugmentedBiGraph import AugmentedBiGraph
from Utils.AugmentedBiGraph import gaussianNLL_F
from Utils.AugmentedBiGraph import gaussianNLL_D

import subprocess
import shlex

import multiprocessing as mp
from  multiprocessing.pool import ThreadPool
from  multiprocessing import Pool

OPTIONS_INIT_UV = ['random', 'exp']

from pygam import LinearGAM, s, f

import logging

class NMF_VB():
    """ Class for structured variational approximation on Assembly Graph"""    
    minW = 1.0e-3
    
    minLogQGamma = 1.0e-100
    minBeta = 1.0e-100
    minVar = 1.0e-3

    
    def __init__(self, prng, X, XN, lengths, G = 2, tauType='fixed', epsilon = 1.0e5, epsilonNoise = 1.0e-3, 
                alpha=1.0e-9,beta=1.0e-9,alpha0=1.0e-9,beta0=1.0e-9, tauThresh = 0.1, 
                maxSampleCov = 0., ARD = True, epsilonPhi=1.0, ARDP=True, BIAS = False, NOISE = False):
                 
        self.prng = prng #random state to store

        #prior parameters for Gamma tau
        self.alpha = alpha
        self.beta  = beta
        self.ARD = ARD
        if self.ARD:
            self.alpha0, self.beta0 = alpha0, beta0
            
        self.BIAS = BIAS
        if self.BIAS:
            self.muTheta0 = muTheta0
            self.tauTheta0 = tauTheta0

        self.tauType = tauType

        self.lengths = lengths

        self.X = X
        
        self.XN = XN
        
        (self.V,self.S) = self.X.shape
        
        self.G = G
        
        self.maxSampleCov = maxSampleCov
        
        self.NOISE = NOISE

        if self.NOISE:
            self.GDash = self.G + 1
            self.epsilonNoise = epsilonNoise
            if self.maxSampleCov > 0.:
                self.epsilonNoise = self.maxSampleCov/self.readLength
        else:
            self.GDash = self.G
            
        self.Omega = self.V*self.S      
 
        #list of mean assignments of strains to graph
        self.expPhi = np.zeros((self.V,self.GDash))
        self.expPhi2 = np.zeros((self.V,self.GDash))
        self.muPhi = np.zeros((self.V,self.GDash))
        self.tauPhi = np.zeros((self.V,self.GDash))
        self.varPhi = np.zeros((self.V,self.GDash))
        
        if self.NOISE:
            self.expPhi[:,self.G] = 1.
            self.expPhi2[:,self.G] = 1.
            self.tauPhi[:,self.G] = 1.
            self.varPhi[:,self.G] = 0.
            self.muPhi[:,self.G] = 1.
            
        self.epsilonPhi = epsilonPhi
        self.ARDP = ARDP
        
        self.epsilon = epsilon #parameter for gamma exponential prior
        self.expGamma = np.zeros((self.GDash,self.S)) #expectation of gamma
        self.expGamma2 = np.zeros((self.GDash,self.S))
        
        self.muGamma = np.zeros((self.GDash,self.S))
        self.tauGamma = np.zeros((self.GDash,self.S))
        self.varGamma = np.zeros((self.GDash,self.S))
        #current excitations on the graph
        self.eLambda = np.zeros((self.V,self.S))
        

        if self.ARD:
            self.alphak_s, self.betak_s = np.zeros(self.G), np.zeros(self.G)
            self.exp_lambdak, self.exp_loglambdak = np.zeros(self.G), np.zeros(self.G)
            for g in range(self.G):
                self.alphak_s[g] = self.alpha0
                self.betak_s[g] = self.beta0
                self.update_exp_lambdak(g)
        
        if self.BIAS:
            self.nBias = self.V
            
            self.biasMap = {v:v for v in range(self.V)}
        
            self.expThetaCat  = np.ones(self.nBias)
            self.expThetaCat.fill(self.muTheta0)
            
            self.expTheta2Cat = np.ones(self.nBias)
            self.expTheta2Cat.fill(self.muTheta0*self.muTheta0)
            
            self.muThetaCat = np.ones(self.nBias)
            self.muThetaCat.fill(self.muTheta0)
            
            self.tauThetaCat = np.ones(self.nBias)
            self.tauThetaCat.fill(self.tauTheta0)
        
        
            self.expTheta  = np.ones(self.V)
            self.expTheta.fill(self.muTheta0)
            
            self.expTheta2 = np.ones(self.V)
            self.expTheta2.fill(self.muTheta0*self.muTheta0)
            
            self.muTheta = np.ones(self.V)
            self.muTheta.fill(self.muTheta0)
            
            self.tauTheta = np.ones(self.V)
            self.tauTheta.fill(self.tauTheta0)

            self.varTheta = 1.0/self.tauTheta 
            
        self.elbo = 0.

        self.logX = np.log(self.X + 0.5)
        self.expTau = np.full((self.V,self.S),self.alpha/self.beta)
        self.expLogTau = np.full((self.V,self.S), digamma(self.alpha)- math.log(self.beta))
        self.betaTau = np.full((self.V,self.S),self.beta)
        self.alphaTau = np.full((self.V,self.S),self.alpha)

        if self.tauType == 'fixed':
            self.bLogTau   = False
            self.bFixedTau = True
            self.bPoissonTau = False
        elif self.tauType == 'log':
            self.bLogTau   = True
            self.bFixedTau = False
            self.bPoissonTau = False
        elif self.tauType == 'empirical':
            self.bLogTau   = False
            self.bFixedTau = False
            self.bPoissonTau = False
        elif self.tauType == 'poisson':
            self.bLogTau   = False
            self.bFixedTau = False
            self.bPoissonTau = True
            
            self.expTau = 1.0/(self.X + 0.5)
            self.expLogTau = np.log(self.expTau)
            
        else:
            print("Hmm... impossible tau strategy disturbing")
            

        self.bLoess = False
        
        self.bGam = True

        self.tauThresh = tauThresh 
        
    def initialise(self,init_UV='exp',mask=None):
        
        if mask is None:
            mask = np.ones((self.V, self.S))
    
        ''' Initialise U, V, tau, and lambda (if ARD). '''
        assert init_UV in OPTIONS_INIT_UV, "Unknown initialisation option: %s. Should be in %s." % (init_UV, OPTIONS_INIT_UV)
        
        
        for v,g in itertools.product(range(self.V),range(self.G)):  
            self.tauPhi[v,g] = 1.
            hyperparam = self.exp_lambdak[g] if self.ARDP else 1.0/self.epsilonPhi
            self.muPhi[v,g] = exponential_draw(hyperparam) if init_UV == 'random' else 1.0/hyperparam
            
            
        for s,g in itertools.product(range(self.S),range(self.GDash)):
            self.tauGamma[g,s] = 1.
            hyperparam = self.exp_lambdak[g] if self.ARD else 1.0/self.epsilon
            self.muGamma[g,s] = exponential_draw(hyperparam) if init_UV == 'random' else 1.0/hyperparam
        
        
        for g in range(self.GDash):
            expGammaG = np.asarray(TN_vector_expectation(self.muGamma[g,:],self.tauGamma[g,:]))
            varGammaG = np.asarray(TN_vector_variance(self.muGamma[g,:],self.tauGamma[g,:]))
            expGamma2G = varGammaG + expGammaG*expGammaG

            self.expGamma[g,:]  = expGammaG
            self.expGamma2[g,:] = expGamma2G
            self.varGamma[g,:]  = varGammaG
        
        for g in range(self.G):
            expPhiG = np.asarray(TN_vector_expectation(self.muPhi[:,g],self.tauPhi[:,g]))
            varPhiG = np.asarray(TN_vector_variance(self.muPhi[:,g],self.tauPhi[:,g]))
            expPhi2G = varPhiG + expPhiG*expPhiG

            self.expPhi[:,g]  = expPhiG
            self.expPhi2[:,g] = expPhi2G
            self.varPhi[:,g]  = varPhiG
        
        
        
        for g in range(self.G):
            self.updatePhi(g,mask)
        for g in range(self.GDash):
            self.updateGamma(g,mask)


        self.eLambda = np.zeros((self.V,self.S))
        for g in range(self.GDash):
            self.addGamma(g)
            
        self.updateTau(True, mask)
                       
        if self.BIAS:
            self.updateTheta(mask)
                
       
    def update_lambdak(self,k):   
        ''' Parameter updates lambdak. '''
        self.alphak_s[k] = self.alpha0 + self.S
        self.betak_s[k] = self.beta0 + self.expGamma[k,:].sum()
    
    def update_exp_lambdak(self,g):
        ''' Update expectation lambdak. '''
        self.exp_lambdak[g] = self.alphak_s[g]/self.betak_s[g]
        self.exp_loglambdak[g] = digamma(self.alphak_s[g]) - math.log(self.betak_s[g])
    

    def removeGamma(self,g_idx):
        
        meanAss = self.expPhi[:,g_idx]
        gammaG  = self.expGamma[g_idx,:]
        
        self.eLambda -= meanAss[:,np.newaxis]*gammaG[np.newaxis,:]

    def addGamma(self,g_idx):
        
        meanAss = self.expPhi[:,g_idx]
        gammaG  = self.expGamma[g_idx,:]
        
        self.eLambda += meanAss[:,np.newaxis]*gammaG[np.newaxis,:]


    def updateTheta(self, mask = None):
        
        assert self.BIAS
        
        if mask is None:
            mask = np.ones((self.V, self.S))
        
        self.eLambda = np.dot(self.expPhi, self.expGamma)
        
        denom = np.sum(self.expTau*self.exp_square_lambda_matrix()*mask,axis=1)*self.lengths*self.lengths  
        
        numer =  self.lengths*np.sum(self.X*self.eLambda*self.expTau*mask,axis=1) 
        
        self.muThetaCat.fill(self.muTheta0*self.tauTheta0)
        self.tauThetaCat.fill(self.tauTheta0)
         
        for v in range(self.V):
            b = self.biasMap[v]
            
            self.muThetaCat[b] += numer[v]
            
            self.tauThetaCat[b] += denom[v]
            
        self.muThetaCat = self.muThetaCat/self.tauThetaCat
    
        self.expThetaCat = np.asarray(TN_vector_expectation(self.muThetaCat,self.tauThetaCat))
        
        self.varThetaCat = np.asarray(TN_vector_variance(self.muThetaCat,self.tauThetaCat))

        self.expTheta2Cat = self.varThetaCat + self.expThetaCat*self.expThetaCat
        
        for v in range(self.V):
            b = self.biasMap[v]
        
            self.muTheta[v] = self.muThetaCat[b]
            
            self.expTheta[v] =  self.expThetaCat[b]
        
            self.varTheta[v] = self.varThetaCat[b]

            self.expTheta2[v] = self.expTheta2Cat[b]

    def updatePhi(self, g_idx, mask = None):
        
        assert g_idx != self.G 
        
        if mask is None:
            mask = np.ones((self.V, self.S))
    
        lamb = 1.0/self.epsilonPhi
        if self.ARDP:
            lamb = self.exp_lambdak[g_idx]    
       
        ''' Parameter updates U. '''   

        if not self.BIAS:       
            temp1 = self.expTau*self.lengths[:,np.newaxis]*self.lengths[:,np.newaxis]
        else:
            temp1 = self.expTau*self.lengths[:,np.newaxis]*self.lengths[:,np.newaxis]*self.expTheta2[:,np.newaxis]   
       
        tauPhiG = np.dot(temp1*mask, self.expGamma2[g_idx,:]) 

        tPhi   = np.delete(self.expPhi,g_idx,1)
        tGamma = np.delete(self.expGamma,g_idx,0)
        #import ipdb; ipdb.set_trace()
        currNELambda = np.dot(tPhi,tGamma)

        if self.BIAS:       
            t1 = self.X*self.expTheta[:,np.newaxis] - currNELambda*self.expTheta2[:,np.newaxis]*self.lengths[:,np.newaxis]
        else:
            t1 = self.X - currNELambda*self.lengths[:,np.newaxis]
    
        t2 = self.expTau*self.lengths[:,np.newaxis]*t1

        muPhiG = 1.0/tauPhiG*(-lamb + np.dot(mask*t2,self.expGamma[g_idx,:]))
        
        expPhiG = np.asarray(TN_vector_expectation(muPhiG,tauPhiG))
        
        varPhiG = np.asarray(TN_vector_variance(muPhiG,tauPhiG))

        expPhi2G = varPhiG + expPhiG*expPhiG

        self.expPhi[:,g_idx] = expPhiG
        
        self.expPhi2[:,g_idx] = expPhi2G

        self.muPhi[:,g_idx] = muPhiG

        self.tauPhi[:,g_idx] = tauPhiG

        self.varPhi[:,g_idx] = varPhiG

    def updateGamma(self, g_idx, mask = None):
        
        if mask is None:
            mask = np.ones((self.V, self.S))
        
        
        temp = np.delete(self.expGamma,g_idx,0)
        temp2 = np.delete(self.expPhi,g_idx,1)
       
        if not self.BIAS:       
            numer = (self.X - np.dot(temp2,temp)*self.lengths[:,np.newaxis])
        else:
            numer = (self.X*self.expTheta[:,np.newaxis] - np.dot(temp2,temp)*self.lengths[:,np.newaxis]*self.expTheta2[:,np.newaxis])   
        
        
        gphi = self.expPhi[:,g_idx]*self.lengths
        
        numer = gphi[:,np.newaxis]*numer

        denom = self.lengths*self.lengths*self.expPhi2[:,g_idx]#dimensions of V
        if self.BIAS:
            denom *= self.expTheta2
                
        dSum = np.dot((self.expTau*mask).transpose(),denom)
        
        numer=numer*self.expTau*mask
        nSum = np.sum(numer,0)
        
        if self.NOISE and g_idx == self.G:
            lamb = 1.0/self.epsilonNoise
        else:
            lamb = 1.0/self.epsilon
            if self.ARD:
                lamb = self.exp_lambdak[g_idx] 
        
        
        nSum -= lamb

        muGammaG = nSum/dSum  
        tauGammaG = dSum

        expGammaG = np.asarray(TN_vector_expectation(muGammaG,tauGammaG))
        
        varGammaG = np.asarray(TN_vector_variance(muGammaG,tauGammaG))

        expGamma2G = varGammaG + expGammaG*expGammaG

        self.expGamma[g_idx,:]  = expGammaG
        self.expGamma2[g_idx,:] = expGamma2G
        self.tauGamma[g_idx,:]  = tauGammaG
        self.muGamma[g_idx,:]   = muGammaG
        self.varGamma[g_idx,:]  = varGammaG
    
        
    def updateTau(self,bFit=True, mask = None):
        
        if self.bPoissonTau:
            self.expTau = 1.0/(self.X + 0.5)

            self.expLogTau = np.log(self.expTau)

            return
        
        if mask is None:
            mask = np.ones((self.V, self.S))
        
        if self.bFixedTau:
            self.updateFixedTau(mask)
        else:
            if self.bLogTau:
                self.updateLogTauX(bFit, mask)
            else:
                self.updateEmpTauX(bFit, mask)
    
    
    def updateFixedTau(self, mask = None):
        
        if mask is None:
            mask = np.ones((self.V, self.S))
    
        Omega = float(np.sum(mask))
    
        square_diff_matrix = self.exp_square_diff_matrix()  

        betaTemp = self.beta + 0.5*np.sum(square_diff_matrix*mask)
        
        alphaTemp = self.alpha + 0.5*Omega

        tempTau = alphaTemp/betaTemp
        
        tempLogTau = digamma(alphaTemp) - np.log(betaTemp)

        self.betaTau.fill(betaTemp/Omega)
        
        self.betaTau = mask*self.betaTau

        self.alphaTau.fill(alphaTemp/Omega)
        
        self.alphaTau = mask*self.alphaTau

        self.expTau.fill(tempTau)
        self.expTau = self.expTau*mask
        
        self.expLogTau.fill(tempLogTau)
        self.expLogTau = self.expLogTau*mask

    def updateLogTauX(self,bFit = True, mask = None):
    
        if mask is None:
            mask = np.ones((self.V, self.S))
    
        square_diff_matrix = self.exp_square_diff_matrix()  
           
        mX = np.ma.masked_where(mask==0, self.X)
        
        X1D = np.ma.compressed(mX)

        
        mSDM = np.ma.masked_where(mask==0, square_diff_matrix)
            
        mBetaTau = self.beta*X1D + 0.5*np.ma.compressed(mSDM)
        
        mBetaTau[mBetaTau < NMF_VB.minBeta] = NMF_VB.minBeta
            
        mLogExpTau = digamma(self.alpha + 0.5) - np.log(mBetaTau)
    
        
        
        mXFit = np.ma.masked_where(mask==0, self.X)
        
        X1DFit = np.ma.compressed(mXFit)
        
        mSDMFit = np.ma.masked_where(mask==0, square_diff_matrix)
            
        mBetaTauFit = self.beta*X1DFit + 0.5*np.ma.compressed(mSDMFit)
        
        mBetaTauFit[mBetaTauFit < NMF_VB.minBeta] = NMF_VB.minBeta
            
        mLogExpTauFit = digamma(self.alpha + 0.5) - np.log(mBetaTauFit)
        
        
        try:
            if bFit:
                self.gam = LinearGAM(s(0,n_splines=5,constraints='monotonic_dec')).fit(X1DFit, mLogExpTauFit)
            
                yest_sm = self.gam.predict(X1D)
                
            else:
                print("Attemptimg linear regression")
                    
                model = LinearRegression()
            
                poly_reg = PolynomialFeatures(degree=2)
            
                X_poly = poly_reg.fit_transform(X1DFit.reshape(-1,1))
            
                model.fit(X_poly, mLogExpTauFit)
            
                X_poly_est = poly_reg.fit_transform(X1D.reshape(-1,1))
            
                yest_sm  = model.predict(X_poly_est)
        except ValueError:
            print("Performing fixed tau")
                    
            self.updateFixedTau(mask)
                    
            return
        
        np.place(self.expLogTau, mask == 1, yest_sm)
         
        np.place(self.expTau, mask == 1, np.exp(yest_sm))
    
        np.place(self.betaTau, mask == 1, mBetaTau)


    def updateEmpTauX(self,bFit = True, mask = None):
    
        if mask is None:
            mask = np.ones((self.V, self.S))
    
        square_diff_matrix = self.exp_square_diff_matrix()  
           
        
        
        mXFit = np.ma.masked_where(mask==0, self.X)
        
        X1DFit = np.ma.compressed(mXFit)
        
        logX1DFit = np.log(0.5 + X1DFit)
        
        mSDMFit = np.ma.masked_where(mask==0, square_diff_matrix)
       
        mFitFit = np.ma.compressed(mSDMFit)
     
        logMFitFit = np.log(mFitFit + NMF_VB.minVar)
        
        if bFit:
            try:
                self.gam = LinearGAM(s(0,n_splines=5,constraints='monotonic_inc')).fit(logX1DFit,logMFitFit)
            
            except ValueError:
                print("Performing fixed tau")
                    
                self.updateFixedTau(mask)
                    
                return
        
        mX = np.ma.masked_where(mask==0, self.X)
        
        X1D = np.ma.compressed(mX)
        
        logX1D = np.log(0.5 + X1D)
         
        yest_sm = self.gam.predict(logX1D)

        mBetaTau = self.beta*(X1D + 0.5) + 0.5*np.exp(yest_sm)

        np.place(self.betaTau, mask == 1, mBetaTau)

        mExpTau = (self.alpha + 0.5)/mBetaTau
        
        np.place(self.expTau, mask == 1, mExpTau)
        
        mLogTau = digamma(self.alpha + 0.5) - np.log(mBetaTau) 
        
        np.place(self.expLogTau, mask == 1, mLogTau)
    

    def update(self, maxIter, mask=None, minDiff=1.0e-3):

        if mask is None:
            mask = np.ones((self.V, self.S))
            
        iter = 0
        self.eLambda = np.dot(self.expPhi, self.expGamma)
        
        self.updateTau(True, mask) 
        
        diffElbo = 1.0
        currElbo=self.calc_elbo(mask)
        
        
        logging.info("Iter, G,  Div, DivF, ELBO, Delta_ELBO")

        while iter < 200 or (iter < maxIter and diffElbo > minDiff):
            
            
            for g in range(self.G):
                
                self.updatePhi(g)
                           
            
            if self.ARD:
                for g in range(self.G):
                    self.update_lambdak(g)
                    self.update_exp_lambdak(g)
            
            for g in range(self.GDash):
                self.updateGamma(g, mask)

            self.eLambda = np.zeros((self.V,self.S))
            for g in range(self.GDash):
                self.addGamma(g)
            
            #if iter % 10 == 0:
            self.updateTau(True, mask)
                       
            if self.BIAS:
                self.updateTheta(mask)
            
            total_elbo = self.calc_elbo(mask)
            diffElbo = abs(total_elbo - currElbo) 
            if np.isnan(diffElbo) or math.isinf(diffElbo):
                diffElbo = 1.
 
            currElbo = total_elbo   
            DivF = self.divF(mask)
            Div  = self.div(mask)
            
            if iter % 10 == 0:
                logging.info("%d, %d, %f, %f, %f, %f", iter, self.G, Div, DivF, total_elbo, diffElbo)

            iter += 1
    
    
    
   

    def div(self,M=None):
        
        if M is None:
            M = np.ones((self.V,self.S))  
        
        """Compute divergence of target matrix from its NMF estimate."""
        Va = self.eLambda
        if self.BIAS:
            Va = self.expTheta[:,np.newaxis]*Va
            
        return (M*(np.multiply(self.XN, np.log(elop(self.XN, Va, truediv))) + (Va - self.XN))).sum()

    def divF(self,M=None):
    
        if M is None:
            M = np.ones((self.V,self.S))

        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""
        
        if self.BIAS:
            R = self.expTheta[:,np.newaxis]*self.eLambda - self.XN
        else:
            R = self.eLambda - self.XN
            
        return (M*np.multiply(R, R)).sum()/np.sum(M)

 
    def divF_matrix(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""

        if self.BIAS:
            R = self.expTheta[:,np.newaxis]*self.eLambda - self.XN
        else:
            R = self.eLambda - self.XN

        return np.multiply(R, R)

    def divergenceN(self, XN, Va):
    
        return (np.multiply(XN, np.log(elop(XN, Va, truediv))) - XN + Va).sum()


    def exp_square_lambda(self):
        ''' Compute: sum_s E_q(phi,gamma) [ sum ( Phi_v Gamma_s )^2 ]. '''
        
        eLambda2Sum = self.eLambda*self.eLambda
        
        diagonal = np.dot(self.expPhi*self.expPhi,self.expGamma*self.expGamma)
        
        return np.sum(eLambda2Sum - diagonal + np.dot(self.expPhi2,self.expGamma2), axis = 1)

    def exp_square_lambda_matrix(self):
        ''' Compute: sum_s E_q(phi,gamma) [ sum ( Phi_v Gamma_s )^2 ]. '''
        
        eLambda2Sum = self.eLambda*self.eLambda
        
        diagonal = np.dot(self.expPhi*self.expPhi,self.expGamma*self.expGamma)
        
        return eLambda2Sum - diagonal + np.dot(self.expPhi2,self.expGamma2)
            
    def mean_diff(self):
    
        diff_matrix = self.divF_matrix()
                        
        return np.sum(diff_matrix)/self.V*self.S    

    def exp_square_diff_matrix(self, bNoise = True): 
        ''' Compute: sum_Omega E_q(phi,gamma) [ ( Xvs - L_v Phi_v Gamma_s )^2 ]. '''
        #return (self.M *( ( self.R - numpy.dot(self.exp_U,self.exp_V.T) )**2 + \
        #                  ( numpy.dot(self.var_U+self.exp_U**2, (self.var_V+self.exp_V**2).T) - numpy.dot(self.exp_U**2,(self.exp_V**2).T) ) ) ).sum()
        
        if bNoise:
            tPhi = self.expPhi
            tGamma = self.expGamma
            tPhi2 = self.expPhi2
            tGamma2 = self.expGamma2
            
        else:
            tPhi = self.expPhi[:,0:self.G]
            tPhi2 = self.expPhi2[:,0:self.G]
            
            tGamma = self.expGamma[0:self.G,:]
            tGamma2 = self.expGamma2[0:self.G,:]
             
        tLambda = np.dot(tPhi, tGamma)
        
        if self.BIAS:
            R = self.X - self.lengths[:,np.newaxis]*self.expTheta[:,np.newaxis]*tLambda
        else:
            R = self.X - self.lengths[:,np.newaxis]*tLambda
            
        t1 = np.dot(tPhi*tPhi, tGamma*tGamma)
        
        if self.BIAS:
            eT2 = self.expTheta*self.expTheta
            t1 = eT2[:,np.newaxis]*t1
        
        diff = np.dot(tPhi2,tGamma2) - t1
        L2 = self.lengths*self.lengths

        if self.BIAS:
            diff = np.dot(tPhi2,tGamma2)*self.expTheta2[:,np.newaxis] - t1
        else:
            diff = np.dot(tPhi2,tGamma2) - t1
        
        diff2 = L2[:,np.newaxis]*diff
        
        return R*R + diff2

    def exp_square_diff(self): 
        ''' Compute: sum_Omega E_q(phi,gamma) [ ( Xvs - L_v Phi_v Gamma_s )^2 ]. '''
        #return (self.M *( ( self.R - numpy.dot(self.exp_U,self.exp_V.T) )**2 + \
        #                  ( numpy.dot(self.var_U+self.exp_U**2, (self.var_V+self.exp_V**2).T) - numpy.dot(self.exp_U**2,(self.exp_V**2).T) ) ) ).sum()
        
        if self.BIAS:
            R = self.X - self.lengths[:,np.newaxis]*self.expTheta[:,np.newaxis]*self.eLambda
        else:
            R = self.X - self.lengths[:,np.newaxis]*self.eLambda
        
        t1 = np.dot(self.expPhi*self.expPhi, self.expGamma*self.expGamma)
        
        if self.BIAS:
            eT2 = self.expTheta*self.expTheta
            t1 = eT2[:,np.newaxis]*t1
            
        if self.BIAS:
            diff = np.dot(self.expPhi2,self.expGamma2)*self.expTheta2[:,np.newaxis] - t1
        else:
            diff = np.dot(self.expPhi2,self.expGamma2) - t1
        
        L2 = self.lengths*self.lengths
            
        diff2 = L2[:,np.newaxis]*diff
        
        return np.sum(R*R + diff2)
        
    def calc_expll(self, mask = None):
        
        if mask is None:
            mask = np.ones((self.V,self.S))
        
        total_elbo = 0.
        
        # Log likelihood
        nTOmega = np.sum(mask)               
        total_elbo += 0.5*(np.sum(self.expLogTau*mask) - nTOmega*math.log(2*math.pi)) #first part likelihood
        total_elbo -= 0.5*np.sum(mask*self.expTau*self.exp_square_diff_matrix()) #second part likelihood

        return total_elbo
        
    def calc_expll_poisson(self, mask = None, bNoise = True):
        
        if mask is None:
            mask = np.ones((self.V,self.S))

        total_elbo = 0.
        
        # Log likelihood
        nTOmega = np.sum(mask)    
        
        poissonWeight = 1.0/(self.X + 0.5)
                   
        total_elbo += 0.5*(np.sum(poissonWeight*mask) - nTOmega*math.log(2*math.pi)) #first part likelihood
        
        total_elbo -= 0.5*np.sum(mask*poissonWeight*self.exp_square_diff_matrix(bNoise = bNoise)) #second part likelihood

        return total_elbo
        
    
    def calc_expll_poisson_maximal(self, mask = None):
        
        if mask is None:
            mask = np.ones((self.V,self.S))

        total_elbo = 0.
        
        # Log likelihood
        nTOmega = np.sum(mask)    
        
        poissonWeight = 1.0/(self.X + 0.5)
        
        self.getMaximalUnitigs('Dummy', drop_strain=None,relax_path=False,writeSeq=False)
        
        pPhi = np.zeros((self.V,self.G))
        
        for gene, mapGene in self.mapGeneIdx.items(): 
        
            for g in range(self.G):
                for node in self.paths[gene][g]:
                    v_idx = mapGene[node[:-1]]
                    pPhi[v_idx,g] = 1.
        
        
        R_pred = self.lengths[:,np.newaxis]*np.dot(pPhi, self.expGamma[0:self.G,:])
        
        if self.BIAS:
            R_pred = R_pred*self.expTheta[:,np.newaxis]
        
                   
        total_elbo += 0.5*(np.sum(poissonWeight*mask) - nTOmega*math.log(2*math.pi)) #first part likelihood
        
        diff_matrix = (self.X - R_pred)**2
        
        total_elbo -= 0.5*np.sum(mask*poissonWeight*diff_matrix) #second part likelihood

        return total_elbo
    

    def calc_elbo(self, mask = None):
    
        if mask is None:
            mask = np.ones((self.V,self.S))

    
        ''' Compute the ELBO. '''
        total_elbo = 0.
        

        # Log likelihood
        nTOmega = np.sum(mask)               
        total_elbo += 0.5*(np.sum(self.expLogTau*mask) - nTOmega*math.log(2*math.pi)) #first part likelihood
        total_elbo -= 0.5*np.sum(mask*self.expTau*self.exp_square_diff_matrix()) #second part likelihood

        if self.NOISE:
            if self.ARD:
            
                total_elbo += self.alpha0 * math.log(self.beta0) - sp.special.gammaln(self.alpha0) \
                          + (self.alpha0 - 1.)*self.exp_loglambdak.sum() - self.beta0 * self.exp_lambdak.sum()
            
                total_elbo += self.S * np.log(self.exp_lambdak).sum() - (self.exp_lambdak[:,np.newaxis] * self.expGamma[0:self.G]).sum()
            
            else:
                total_elbo += np.sum(-math.log(self.epsilon) - self.expGamma[0:self.G]/self.epsilon)
       
            total_elbo += np.sum(-math.log(self.epsilonNoise) - self.expGamma[self.G]/self.epsilonNoise)
        
        else:
            # Prior lambdak, if using ARD, and prior U, V
            if self.ARD:
            
                total_elbo += self.alpha0 * math.log(self.beta0) - sp.special.gammaln(self.alpha0) \
                          + (self.alpha0 - 1.)*self.exp_loglambdak.sum() - self.beta0 * self.exp_lambdak.sum()
            
                total_elbo += self.S * np.log(self.exp_lambdak).sum() - (self.exp_lambdak[:,np.newaxis] * self.expGamma).sum()
            
            else:
                total_elbo += np.sum(-math.log(self.epsilon) - self.expGamma/self.epsilon)
            
        
        #Prior theta if using bias
        
        if self.BIAS:
            dS = np.sqrt(self.tauTheta0/2.0)*self.muTheta0
            thetaConst = 0.5*np.log(self.tauTheta0/(2.0*np.pi)) -0.5*self.tauTheta0*self.muTheta0*self.muTheta0 - np.log(0.5*(1 + erf(dS)))

            lnThetaPrior = self.V*thetaConst
            
            #thetaMoment1 = np.array(TN_vector_expectation(self.expTheta,self.tauTheta))
            #thetaVar =  np.array(TN_vector_variance(self.expTheta,self.tauTheta))
            #thetaMoment2 = thetaVar + 2.0*self.expTheta*thetaMoment1 - self.expTheta*self.expTheta
            lnThetaPrior += np.sum(-0.5*self.tauTheta0*(self.expTheta2 - 2.0*self.expTheta*self.muTheta0))
            total_elbo += lnThetaPrior 
    
        
        #add tau prior
    
        if self.bFixedTau:
            total_elbo += nTOmega*(self.alpha * math.log(self.beta) - sps.gammaln(self.alpha)) 
            total_elbo += np.sum((self.alpha - 1.)*self.expLogTau*mask - self.beta*self.expTau*mask)

        # q for lambdak, if using ARD
        if self.ARD:
            total_elbo += - sum([v1*math.log(v2) for v1,v2 in zip(self.alphak_s,self.betak_s)]) + sum([sp.special.gammaln(v) for v in self.alphak_s]) \
                          - ((self.alphak_s - 1.)*self.exp_loglambdak).sum() + (self.betak_s * self.exp_lambdak).sum()

        #add q for gamma
        qGamma = -0.5*np.log(self.tauGamma).sum() + 0.5*self.GDash*self.S*math.log(2.*math.pi)
        temp = sps.erfc(-self.muGamma*np.sqrt(self.tauGamma)/math.sqrt(2.))
        
        temp[temp < NMF_VB.minLogQGamma] = NMF_VB.minLogQGamma
        

        qGamma += np.log(0.5*temp).sum()
        qGamma += (0.5*self.tauGamma * ( self.varGamma + (self.expGamma - self.muGamma)**2 ) ).sum()

        total_elbo += qGamma
        
        if self.BIAS:
            
            qTheta = -0.5*np.log(self.tauTheta).sum() + 0.5*self.V*math.log(2.*math.pi)
            qTheta += np.log(0.5*sps.erfc(-self.muTheta*np.sqrt(self.tauTheta)/math.sqrt(2.))).sum()
            qTheta += (0.5*self.tauTheta * ( self.varTheta + (self.expTheta - self.muTheta)**2 ) ).sum()
        
            total_elbo += qTheta
        
        # q for tau
        if self.bFixedTau:
            dTemp1 = (self.alpha + 0.5)*np.sum(np.log(self.betaTau)*mask) - nTOmega*sps.gammaln(self.alpha + 0.5)    
            dTemp2 = np.sum((self.alpha - 0.5)*self.expLogTau*mask) + np.sum(self.betaTau*self.expTau*mask)

            total_elbo += - dTemp1 
            total_elbo += - dTemp2
        
        return total_elbo

    def predict(self, M_pred):
        ''' Predict missing values in R. '''
        R_pred = self.lengths[:,np.newaxis]*np.dot(self.expPhi, self.expGamma)
        
        if self.BIAS:
            R_pred = R_pred*self.expTheta[:,np.newaxis]
        
        MSE = self.compute_MSE(M_pred, self.X, R_pred)
        #R2 = self.compute_R2(M_pred, self.R, R_pred)    
        #Rp = self.compute_Rp(M_pred, self.R, R_pred)        
        return MSE

    def predict_sqrt(self, M_pred):
        ''' Predict missing values in R. '''
        R_pred = self.lengths[:,np.newaxis]*np.dot(self.expPhi, self.expGamma)
        
        if self.BIAS:
            R_pred = R_pred*self.expTheta[:,np.newaxis]
        
        MSE = self.compute_MSE(M_pred, np.sqrt(self.X), np.sqrt(R_pred))
        #R2 = self.compute_R2(M_pred, self.R, R_pred)    
        #Rp = self.compute_Rp(M_pred, self.R, R_pred)        
        return MSE


    ''' Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation) '''
    def compute_MSE(self,M,R,R_pred):
        ''' Return the MSE of predictions in R_pred, expected values in R, for the entries in M. '''
        return (M * (R-R_pred)**2).sum() / float(M.sum())





def main(argv):
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    #import ipdb; ipdb.set_trace()

    np.random.seed(2)
    prng = RandomState(238329)
    

 #       print("Debug")
if __name__ == "__main__":
    main(sys.argv[1:])


