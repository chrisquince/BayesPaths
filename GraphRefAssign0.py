from UnitigGraph import UnitigGraph
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
from scipy.sparse.linalg import inv
import math
import argparse
from subprocess import Popen, PIPE, STDOUT
import networkx as nx

import collections
from collections import deque
from collections import defaultdict
from collections import Counter
from numpy.random import RandomState
from scipy.sparse import csc_matrix
import warnings
    
mapBase = {'A': 0, 'C': 1, 'G': 2, 'T': 3} 

mapDirn = {'True' : "+", 'False' : "-"}

def Most_Common(lst):
    
    data = Counter(lst)
    
    return data.most_common(1)[0][0]

def trans(x):

    if x > 0:
        x = x - 1
    else:
        x = x + 1
        
    return x;

def compU(unitigd):

    if unitigd[0] == '-':
        unitig = unitigd[1:]
    else:
        unitig = "-" + unitigd
   
    return unitig  

def removeD(unitigd):

    if unitigd[0] == '-':
        unitig = unitigd[1:]
    else:
        unitig = unitigd
    return unitig

def reverseComplement(unitigList):

    reverseList = list(reversed(unitigList))

    compList = [compU(x) for x in reverseList]

    return compList

def convertNameToNode2(unitigd):
    
    direction = True
    if unitigd[0] == '-':
        unitig = unitigd[1:]
        direction = False
    else:
        unitig = unitigd
          
    return (unitig,direction)

def expNormLogProb(logProbs):

    maxP = np.max(logProbs)
    
    ds = logProbs - maxP

    probs = np.exp(ds)
    
    probs /= probs.sum()
    
    return probs

class Graph_Correct():
    """Creates graph correct object with Laplacian regularisation"""


    def __init__(self,randomState, unitigGraph, theta = 1.0e-6, maxIter = 100, alphaPrior = 1., minP = 1.0, tau = 1.0):
    
        self.randomState = randomState
        
        self.unitigGraph = unitigGraph
        
        self.epsilon = np.zeros((2,4,4))
        
        self.epsilon.fill(0.01/3.)
        
        self.eta = np.zeros((2,4,4))
        
        self.theta = theta
        
        self.maxIter = maxIter
        
        self.alphaPrior = 1.
        
        self.minP = minP
        
        self.tau = tau
        
        np.fill_diagonal(self.epsilon[0,:],0.99)
        np.fill_diagonal(self.epsilon[1,:],0.99)
        
 
        self.unitigs = list(self.unitigGraph.undirectedUnitigGraph.nodes())
        self.unitigs.sort(key=int)
        
        self.unitigMap = {j:i for i,j in enumerate(self.unitigs)}
        
        self.NNodes = len(self.unitigs)
        
        self.overlapEnd = {}
        self.adjLengths = {}
        self.NC = {}
        for unitig in self.unitigs:
            adjLength = self.unitigGraph.lengths[unitig]

            nC = len(self.unitigGraph.undirectedUnitigGraph.neighbors(unitig))
        
            assert adjLength >= 0
            
            self.adjLengths[unitig] = adjLength
            self.NC[unitig] = nC
    def readPaths(self,pathFile):
    
        self.pathMap = defaultdict(list)

        self.forward = {}
        degenerateMap = defaultdict(lambda : defaultdict(list))
    
        with open(pathFile) as f:
            
            for line in f:
                line = line.rstrip()
                
                m = re.match(r">(\S+)", line)
                
                if m:
                    read_id = m.group(1)
                    
                    n = re.match(r"\S+/(1|2)", read_id)
                    
                    forward = 0
                    if n:
                        dirn_id = n.group(1)
                        if dirn_id == '1':
                            forward  = 0
                        else:
                            forward = 1
                    self.forward[read_id] = forward       
                    nextline = next(f)
                    nextline = nextline.rstrip()
                    path = nextline.split(';')
                    path = list(filter(None, path))
                    
                    endPos = int(path.pop())
                    startPos = int(path.pop())
                    
                    path = [int(x) for x in path]
                    
                    path = [trans(x) for x in path]
                    
                    path = [str(x) for x in path]
                    
                    unitigs = [removeD(x) for x in path]
                    
                    #deal with case where we only hit one unitis
                    
                    pathC = reverseComplement(path)
                    
                    pathU = ';'.join(path)
                    pathUC = ';'.join(pathC)
                    if pathUC > pathU:
                        pathU = pathUC
                    
                    pathR = [convertNameToNode2(x) for x in path]


                    nextline = next(f)
                    nextline = nextline.rstrip()
                    readseq = nextline
                    nextline = next(f)
                    nextline = nextline.rstrip()
                    pathseq = nextline
                    
                    alignMatrix = np.zeros((4,4),dtype=np.int) 
                    
                    for align in zip(readseq, pathseq):
                        baseA = mapBase[align[0]]
                        baseB = mapBase[align[1]]
                        alignMatrix[baseA,baseB] += 1
                    mismatch = np.sum(alignMatrix) - np.trace(alignMatrix)
                    
                    alignMatrix[0][1] + alignMatrix[0][2] + alignMatrix[0][1]
                    degenerateMap[read_id][pathseq].append([pathR,mismatch, alignMatrix, pathU, pathseq, unitigs, startPos, endPos])
            
            self.nodeSum = {} 
            self.readZ   = {}  
            for read, maps in degenerateMap.items():

                for path, aligns in maps.items():
                    
                    aligns.sort(key=lambda x:len(x[0]))
                    
                    self.pathMap[read].append(aligns[0])
                    
                    
            for read, aligns in self.pathMap.items():
                self.readZ[read] = np.zeros(len(aligns))
                for align in aligns:
                    unitigs = align[5]
                    startPos = align[6]
                    endPos = align[7]
                
                    for unitig in unitigs:
                        if unitig not in self.nodeSum:
                            self.nodeSum[unitig] = 0.0
                
                    fHits = np.ones(len(unitigs))
                    if len(unitigs) == 1:
                        
                        fHits[0] = float(startPos - endPos)/self.adjLengths[unitigs[0]]
                        
                    else:
                    
                        #decide whether to trip start of read
                        startUnitig = unitigs[0]

                        fHits[0] =  float(self.adjLengths[startUnitig] - startPos)/float(self.adjLengths[startUnitig])
                        
                        endUnitig = unitigs[-1]
 
                        fHits[-1] =  float(endPos)/float(self.adjLengths[endUnitig])
                        
                    align.append(fHits)
            
            self.NNodes = len(self.nodeSum)
            self.thetaDash = self.theta/self.NNodes
    
    def updateEpsilon(self):
    
        for r in (0,1):
            for e in range(4):
                alpha = self.eta[r,e,:] + self.alphaPrior
                self.epsilon[r,e,:] =  self.randomState.dirichlet(alpha)  


    def updateZ(self):
        nchange = 0
        # degenerateMap[read_id][pathseq].append([pathR,mismatch, alignMatrix, pathU, pathseq, unitigs, startPos, endPos, fHits])
            
        #now reassign based on Dirichlet across graph 
        for read, aligns in self.pathMap.items():
            r = self.forward[read]
            NAligns = len(aligns)
            logEpsilon = np.log(self.epsilon[r,:,:])
            
            if NAligns > 1:
                #remove current assignment
               
                alignCurr = self.readAssign[read]
                graphCurr = alignCurr[3]
                self.graphFreq[graphCurr] -= 1
                for (unitig,f) in zip(alignCurr[5],alignCurr[8]):
                    self.nodeSum[unitig] -= f
                
                self.eta[r,:,:] -= alignCurr[2]
                
                logP = np.zeros(NAligns)
            
                i = 0
                for align in aligns:
                    graphA = align[3]
                    unitigs = align[5]
                    
                    emission = [self.nodeSum[x] for x in unitigs]
                    
                    F =  min(emission) + self.thetaDash
                    
                    assert F >= 0.0
                        
                    logP[i] = np.log(F) + np.sum(align[2]*logEpsilon)
                    i=i+1
            
                P = expNormLogProb(logP)
                
                z = self.randomState.choice(NAligns, 1, p=P)
                self.eta[r,:,:] += aligns[z[0]][2]
                alignNew = aligns[z[0]]
                graphNew = alignNew[3]
                self.graphFreq[graphNew] += 1
                self.readAssign[read] = alignNew
                for (unitig,f) in zip(alignNew[5],alignNew[8]):
                    self.nodeSum[unitig] += f
                    
                if graphCurr != graphNew:
                    nchange += 1
        return nchange


                
    def finalAssign(self,outFile):

        
        for graphNode in self.meanSum:
            if self.meanSum[graphNode] < self.minP:
                self.meanSum[graphNode] = 0.
            
        
        #now reassign based on Dirichlet across graph but only to parts with sufficient logProb
        
        with open(outFile, mode='w') as file:    
            for read, aligns in self.pathMap.items():
                NAligns = len(aligns)
                r = self.forward[read]
                logEpsilon = np.log(self.epsilon[r,:,:])
                print("Assigning " + read)
                minE = []
                goodAligns = []
                
                for align in aligns:

                    unitigs = align[5]
                    
                    emission = [self.meanSum[x] for x in unitigs]
                    
                    F =  min(emission) 
                    if F > 0.0:
                        goodAligns.append(align)
                        minE.append(F)
                            
                
                NGAligns = len(goodAligns)
                graphCurr = self.readAssign[read][3]
                self.graphFreq[graphCurr] -= 1
                
                if NGAligns >= 1:
                    
                    logP = np.zeros(NGAligns)

                    i = 0
                    for align in goodAligns:
                        graphA = align[3]
                        unitigs = align[5]
                    
                        F =  minE[i]

                        logP[i] = np.log(F) + np.sum(align[2]*logEpsilon)
         
                        i=i+1
                        
                    P = expNormLogProb(logP)
                
                    z = self.randomState.choice(NGAligns, 1, p=P)

                    alignNew = goodAligns[z[0]]
                    self.graphFreq[alignNew[3]] += 1
                    self.readAssign[read] = alignNew
                    print("Assigned to " + alignNew[3])
                    file.write(">"+read + "\n")
                    file.write(alignNew[4] + "\n")
                else:
                    print("Removed read " + read)
                    
    def assignReads(self):
    
        self.readAssign = {}
        self.graphFreq = defaultdict(int)
        #assign reads to best match to begin with 
        for read, aligns in self.pathMap.items():
            
            aligns.sort(key=lambda x:x[1])
            
            for (unitig,f) in zip(aligns[0][5],aligns[0][8]):
                assert f <= 1.0
                self.nodeSum[unitig] += f
            
            r = self.forward[read]
            self.readAssign[read] = aligns[0]
            self.eta[r,:,:] += aligns[0][2]
            self.graphFreq[aligns[0][3]] += 1       
                
        iter = 0
        while iter < self.maxIter:
            self.updateEpsilon()
        
            nchange = self.updateZ()
            
            iter = iter + 1
            print(str(iter) + "," + str(nchange))
        
        iter = 0
        self.meanSum = defaultdict(float)
        while iter < self.maxIter:
            self.updateEpsilon()
        
            nchange = self.updateZ()
            
            for graphNode in self.nodeSum:
                self.meanSum[graphNode] += self.nodeSum[graphNode]
            print(str(iter) + "," + str(nchange))
            iter = iter + 1

        for graphNode in self.meanSum:
            self.meanSum[graphNode] /= float(self.maxIter)
            



def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("unitig_file", help="unitig fasta file in Bcalm2 format")

    parser.add_argument("kmer_length", help="kmer length assumed overlap")

    parser.add_argument("path_file", help="bgreat path file")

    parser.add_argument("blast_file", help="blast m8 reads against COG reference")

    parser.add_argument('-s','--samples',nargs='?', default=1, type=int, 
        help=("number of samples in data set"))

    parser.add_argument('-hmm', action='store_true',help=("hmm input"))

    args = parser.parse_args()

    import ipdb; ipdb.set_trace()

    np.random.seed(2)
    prng = RandomState(238329)

    unitigGraph = UnitigGraph.loadGraph(args.unitig_file, int(args.kmer_length))

    graphCorrect = Graph_Correct(prng, unitigGraph)
    
    graphCorrect.readPaths(args.path_file)
        
    readHits = {}
    with open(args.blast_file) as f:
        for line in f:
            line = line.rstrip()
            #S25_NC_005791.1-23512/1 rna_hmm3        rRNA    1       149     1.8e-42 +       NA      16S_rRNA

            if args.hmm:
                (read,hit,hmmType,qstart,qend,evalue,strand,notsure,model) = line.split('\t')
            else:
                (read,hit,pident,alignlength,mis,gap,qstart,qend,subjstart,subjend,evalue,bitscore) = line.split('\t')
            
            if args.hmm:
                forward = strand
                subjstart = 0
                subjend = 0
            else:
                forward = True
            
                if qstart >= qend:
                    forward = False
                 
            readHits[read] = (int(subjstart),int(subjend),forward)
            
            #S0_NZ_CP009045.1-429776/1       gi|384199956|ref|YP_005585699.1|        79.6    49      10      0       2       148     69      117     1.2e-13 77.8
    
    
    nodeStart = {}
    nodeEnd = {}
    nodeCov = defaultdict(lambda: np.zeros(args.samples))
    
    nodeOrient = defaultdict(list)
    for read, aligns in graphCorrect.pathMap.items():
        aligns.sort(key=lambda x:x[1])
        
        bestPath = aligns[0][0]
        
        if read in readHits:
            (subjstart,subjend,forward) = readHits[read] 
        
            
            #readToks = read.split("_")
            sample_idx = 0 #int(readToks[0][1:])

            if forward:
                startNode = bestPath[0]
                startDirn = startNode[1]
                node = startNode[0]

                if node in nodeStart:
                    if subjstart < nodeStart[node]:
                        nodeStart[node] = subjstart
                else:
                    nodeStart[node] = subjstart
                
                nodeOrient[node].append(startDirn)
                    
                endNode = bestPath[-1]
                endDirn = endNode[1]
                node = endNode[0]

                if node in nodeEnd:
                    if subjend > nodeEnd[node]:
                        nodeEnd[node] = subjend
                else:
                    nodeEnd[node] = subjend
                
                nodeOrient[node].append(endDirn)
            
            else:
                startNode = bestPath[-1]
                startDirn = startNode[1]
                node = startNode[0]

                if node in nodeStart:
                    if subjstart < nodeStart[node]:
                        nodeStart[node] = subjstart
                else:
                    nodeStart[node] = subjstart
                
                nodeOrient[node].append(not startDirn)
                
                endNode = bestPath[0]
                endDirn = endNode[1]
                node = endNode[0]

                if node in nodeEnd:
                    if subjend > nodeEnd[node]:
                        nodeEnd[node] = subjend
                else:
                    nodeEnd[node] = subjend
                
                nodeOrient[node].append(not endDirn)  
        
        n = 0
        for node in bestPath:
            fracHit = aligns[0][8][n]
            nodeCov[node[0]][sample_idx] += fracHit
            n += 1
    sortUnitigs = list(unitigGraph.undirectedUnitigGraph.nodes())
    sortUnitigs.sort(key=int)
    for unitig in sortUnitigs:
    
        if unitig in nodeStart:
            start = nodeStart[unitig]
        else:
            start = None

        if unitig in nodeEnd:
            end = nodeEnd[unitig]
        else:
            end = None
        
        if len(nodeOrient[unitig]) > 0:
            orient = Most_Common(nodeOrient[unitig])
        else:
            orient = None
            
        print(unitig + "," + str(start) + "," + str(end) + "," + str(orient))

    with open('Coverage.csv', 'w') as cov_file:
    
        for unitig in sortUnitigs:

            if unitig in nodeCov:
                covs = nodeCov[unitig]
            else:
                covs = np.zeros(args.samples)
            covList = covs.tolist()
            fCovList = [str(x) for x in covList] 
            cov_file.write(unitig + "," + ",".join(fCovList)+"\n") 

            
    #graphCorrect.assignReads()
    
    #print(graphCorrect.epsilon[0,:,:])
    #print(graphCorrect.epsilon[1,:,:])
    
    #graphCorrect.finalAssign(args.corrected_read_file)
    
if __name__ == "__main__":
    main(sys.argv[1:])
