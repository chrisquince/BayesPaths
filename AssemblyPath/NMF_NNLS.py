import numpy as np
import math

from scipy.optimize import nnls

OPTIONS_INIT_PG = ['ones', 'random', 'exponential']
MAX_NNLS_ITER = 1000
MED_MIN_Q=0.25
MIN_DELTA=1.0e-6

class NMF_NNLS:
    def __init__(self,X,M,G):
        ''' Set up the class and do some checks on the values passed. '''
        self.X = np.array(X,dtype=float)
        self.M = np.array(M,dtype=float)
        self.G = G                     
                
        assert len(self.X.shape) == 2, "Input matrix R is not a two-dimensional array, " \
            "but instead %s-dimensional." % len(self.X.shape)
        assert self.X.shape == self.M.shape, "Input matrix R is not of the same size as " \
            "the indicator matrix M: %s and %s respectively." % (self.X.shape,self.M.shape)
            
        (self.V,self.S) = self.X.shape
        
        self.check_empty_rows_columns() 
                 
      
    def check_empty_rows_columns(self):
        ''' Raise an exception if an entire row or column is empty. '''
        sums_columns = self.M.sum(axis=0)
        sums_rows = self.M.sum(axis=1)
                    
        # Assert none of the rows or columns are entirely unknown values
        for i,c in enumerate(sums_rows):
            assert c != 0, "Fully unobserved row in R, row %s." % i
        for j,c in enumerate(sums_columns):
            assert c != 0, "Fully unobserved column in R, column %s." % j
            
        
    
    def train(self,iterations,no_runs = 10, init_PG='random',expo_prior=1.):
        ''' Initialise and run the algorithm. '''
        
        bestR2 = 0.
        bestP = None
        bestGa = None
        
        for n in range(no_runs):
            self.initialise(init_PG=init_PG,expo_prior=expo_prior)
         
            R2 = self.run(iterations=iterations)   
        
            if R2 > bestR2:
                bestP = self.P
                bestGa = self.Ga
    
        print('Best R2: ' + str(R2))
        self.P = bestP
        self.Ga = bestGa
    
    
    def initialise(self,init_PG='random',expo_prior=1.):
        ''' Initialise U and V. '''
        assert init_PG in OPTIONS_INIT_PG, "Unrecognised init option for U,V: %s. Should be one in %s." % (init_PG, OPTIONS_INIT_PG)
        
        if init_PG == 'ones':
            self.P = np.ones((self.V,self.G))
            self.Ga = np.ones((self.G,self.S))
        elif init_PG == 'random':
            self.P = np.random.rand(self.V,self.G)
            self.Ga = np.random.rand(self.G,self.S)
        elif init_PG == 'exponential':
            self.P = np.random.exponential(scale=expo_prior, size=(self.V,self.G))
            self.Ga = np.random.exponential(scale=expo_prior, size=(self.G,self.S))
    
    
    def run(self,iterations):
        ''' Run the algorithm. '''
        assert hasattr(self,'P') and hasattr(self,'Ga'), "P and Ga have not been initialised - please run NMF.initialise() first."        
        

        it = 0
        
        Delta = MIN_DELTA + 1
        lastR2 = -1.
        while it < iterations and Delta > MIN_DELTA:
            for v in range(self.V):
                self.update_P(v)
            for s in range(self.S):
                self.update_Ga(s)
            
            perf = self.give_update(it)
            
            Delta = abs(perf['R^2'] - lastR2)
            
            lastR2 = perf['R^2']
            
            it += 1
        
        return lastR2     
            
    def update_Ga(self,s):
        ''' Update values for Gamma[,s]. '''
        
        PM = self.P[self.M[:,s] > 0,:]
        XM = self.X[self.M[:,s] > 0,s]
        
        self.Ga[:,s] = nnls(PM, XM)[0]
    
    def update_P(self,v):
        ''' Update values for Gamma[,s]. '''
        
        GTM = np.transpose(self.Ga)[self.M[v,:] > 0,:]
        XTM = np.transpose(self.X)[self.M[v,:] > 0,v]
        
        self.P[v,:] = nnls(GTM, XTM)[0]
    
    def factorizeG(self, iterations):

       
        for g in range(self.G):
            Wg = self.P[:,g]
            qW = np.quantile(Wg, MED_MIN_Q)
             
            Wg[Wg <= qW] = 1.0e-10
            Wg[Wg > qW]  = 1.0
            
            self.P[:,g] = Wg
        

        for s in range(self.S):
            self.update_Ga(s)
                
        self.give_update(-1)
                
    
    def factorizeP(self,iterations):
        for v in range(self.V):
            self.update_P(v)
            
        self.give_update(-2)

    
    def predict(self,M_pred):
        ''' Predict missing values in R. '''
        R_pred = np.dot(self.P,self.Ga)
        MSE = self.compute_MSE(M_pred,self.X,R_pred)
        R2 = self.compute_R2(M_pred,self.X,R_pred)    
        Rp = self.compute_Rp(M_pred,self.X,R_pred)        
        return {'MSE':MSE,'R^2':R2,'Rp':Rp}   
    
    
    ''' Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation) '''
    def compute_MSE(self,M,R,R_pred):
        ''' Return the MSE of predictions in R_pred, expected values in R, for the entries in M. '''
        return (M * (R-R_pred)**2).sum() / float(M.sum())
        
    def compute_R2(self,M,R,R_pred):
        ''' Return the R^2 of predictions in R_pred, expected values in R, for the entries in M. '''
        mean = (M*R).sum() / float(M.sum())
        SS_total = float((M*(R-mean)**2).sum())
        SS_res = float((M*(R-R_pred)**2).sum())
        return 1. - SS_res / SS_total if SS_total != 0. else np.inf
        
    def compute_Rp(self,M,R,R_pred):
        ''' Return the Rp of predictions in R_pred, expected values in R, for the entries in M. '''
        mean_real = (M*R).sum() / float(M.sum())
        mean_pred = (M*R_pred).sum() / float(M.sum())
        covariance = (M*(R-mean_real)*(R_pred-mean_pred)).sum()
        variance_real = (M*(R-mean_real)**2).sum()
        variance_pred = (M*(R_pred-mean_pred)**2).sum()
        return covariance / float(math.sqrt(variance_real)*math.sqrt(variance_pred))   
               
        
    def give_update(self,iteration):    
        ''' Print and store the I-divergence and performances. '''
        perf = self.predict(self.M)
               
        print ("Iteration %s. MSE: %s. R^2: %s. Rp: %s." % (iteration,perf['MSE'],perf['R^2'],perf['Rp']))
        
        return perf