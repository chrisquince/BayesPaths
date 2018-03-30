import numpy as np
import math
from operator import mul, truediv, eq, ne, add, ge, le, itemgetter


mapDirn = {'True' : "+", 'False' : "-"}

complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N' : 'N'} 

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

def read_coverage_file(coverage_file):

    covMap = {}  
    with open(coverage_file) as f:
        for line in f:
            line = line.rstrip()
            
            tokens = line.split(',')
            
            idx = tokens[0]
            
            tokens.pop(0)
            
            covMap[idx] = np.asarray([float(t) for t in tokens],dtype=np.float)

    return covMap

def reverseComplement(seq):
    bases = list(seq) 
    bases = reversed([complement.get(base,base) for base in bases])
    bases = ''.join(bases)
    return bases
