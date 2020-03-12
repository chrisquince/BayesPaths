import numpy as np
import math
from scipy.stats import truncnorm, norm
from scipy.special import erfc
from collections import defaultdict
from operator import mul, truediv, eq, ne, add, ge, le, itemgetter


mapDirn = {'True' : "+", 'False' : "-"}

complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N' : 'N'} 

# TN expectation    
def TN_vector_expectation(mus,taus):
    sigmas = np.float64(1.0) / np.sqrt(taus)
    x = - np.float64(mus) / sigmas
    lambdax = norm.pdf(x)/(0.5*erfc(x/math.sqrt(2)))
    exp = mus + sigmas * lambdax
    
    # Exp expectation - overwrite value if mu < -30*sigma
    exp = [1./(np.abs(mu)*tau) if mu < -30 * sigma else v for v,mu,tau,sigma in zip(exp,mus,taus,sigmas)]
    return [v if (v >= 0.0 and v != np.inf and v != -np.inf and not np.isnan(v)) else 0. for v in exp]
    
# TN variance
def TN_vector_variance(mus,taus):
    sigmas = np.float64(1.0) / np.sqrt(taus)
    x = - np.float64(mus) / sigmas
    lambdax = norm.pdf(x)/(0.5*erfc(x/math.sqrt(2)))
    deltax = lambdax*(lambdax-x)
    var = sigmas**2 * ( 1 - deltax )
    
    # Exp variance - overwrite value if mu < -30*sigma
    var = [(1./(np.abs(mu)*tau))**2 if mu < -30 * sigma else v for v,mu,tau,sigma in zip(var,mus,taus,sigmas)]
    return [v if (v >= 0.0 and v != np.inf and v != -np.inf and not np.isnan(v)) else 0. for v in var]      

              
# TN expectation        
def TN_expectation(mu,tau):
    sigma = np.float64(1.0) / math.sqrt(tau)
    if mu < -30 * sigma:
        exp = 1./(abs(mu)*tau)
    else:
        x = - mu / sigma
        lambdax = norm.pdf(x)/(0.5*erfc(x/math.sqrt(2)))
        exp = mu + sigma * lambdax
    return exp if (exp >= 0.0 and exp != numpy.inf and exp != -numpy.inf and not numpy.isnan(exp)) else 0.
       
# TN variance
def TN_variance(mu,tau):
    sigma = np.float64(1.0) / math.sqrt(tau)
    if mu < -30 * sigma:
        var = (1./(abs(mu)*tau))**2
    else:
        x = - mu / sigma
        lambdax = norm.pdf(x)/(0.5*erfc(x/math.sqrt(2)))
        deltax = lambdax*(lambdax-x)
        var = sigma**2 * ( 1 - deltax )
    return var if (var >= 0.0 and var != np.inf and var != -np.inf and not np.isnan(var)) else 0.       
       
# TN mode
def TN_mode(mu):
    return max(0.0,mu)

def readRefHits(refHitFile):
    refHits = defaultdict(dict)
    allHits = set()
    
    with open(refHitFile) as f:
        for line in f:
            line = line.rstrip()
            tokens = line.split('\t')
            unitig = tokens[0]
            
            hit = tokens[1]
            pid = tokens[2]
            div = tokens[4]
            
            refHits[unitig][hit] = float(div)
            if hit not in allHits:
                allHits.add(hit)
    return (refHits,allHits)

def readRefAssign(refAssignFile):
    refHits = defaultdict(dict)
    allHits = set()

    with open(refAssignFile) as f:
        for line in f:
            line = line.rstrip()
            tokens = line.split('\t')
            unitig = tokens[0]

            tokens.pop(0)

            for ref in tokens:
                refHits[unitig][ref] = 0.
            if ref not in allHits:
                allHits.add(ref)
    return (refHits,allHits)


def elop(Xt, Yt, op):
    X = np.copy(Xt)
    Y = np.copy(Yt)
    try:
        X[X == 0] = np.finfo(X.dtype).eps
        Y[Y == 0] = np.finfo(Y.dtype).eps
    except ValueError:
        return op(X, Y)
    return op(X, Y)

def convertNodeToName(node_code):
    
    if all(isinstance(item, tuple) for item in node_code):
        tempList =  [element for tupl in node_code for element in tupl]
    else:
        tempList = [element for element in node_code]
    nodeName = [mapDirn[str(w)] if str(w) in mapDirn else str(w) for w in tempList]
        
    return "".join(nodeName)
    
    
def convertNameToNode2(unitigd):
    unitig = unitigd[:-1]
    dirn = unitigd[-1:] 
            
    if dirn == '+':
        direction = True
    else:
        direction = False
            
    return (unitig,direction)    

def expNormLogProb(logProbs):

    maxP = np.max(logProbs)
    
    ds = logProbs - maxP

    probs = np.exp(ds)
    
    probs /= probs.sum()
    
    return probs

def expLogProb(logProbs):

    maxP = np.max(logProbs)
    
    ds = logProbs - maxP

    probs = np.exp(ds)
    
    return (probs, maxP)


def read_unitig_order_file(unitig_order_file):
    """Read unitig directions"""
    
    unitig_order = {}
    
    with open(unitig_order_file) as f:
        for line in f:
        
            line = line.rstrip()
        
            tokens = line.split(',')
        
            unitig = tokens[0]
            
            if tokens[1] != "None":
                start = int(tokens[1])
            else:
                start = None
                
            if tokens[2] != "None":
                end = int(tokens[2])
            else:
                end = None
                
            if tokens[3] == "True":
                dirn = True
            else:
                dirn = False
            
            unitig_order[unitig] = (start,end,dirn)
         
    return unitig_order

def read_coverage_file(coverage_file, tsvFile=False):

    if tsvFile:
        sep = '\t'
    else:
        sep = ','

    covMap = {}  
    with open(coverage_file) as f:
        for line in f:
            line = line.rstrip()
            
            tokens = line.split(sep)
            
            idx = tokens[0]
            
            tokens.pop(0)
            
            covMap[idx] = np.asarray([float(t) for t in tokens],dtype=np.float)

    return covMap

def reverseComplement(seq):
    bases = list(seq) 
    bases = reversed([complement.get(base,base) for base in bases])
    bases = ''.join(bases)
    return bases
