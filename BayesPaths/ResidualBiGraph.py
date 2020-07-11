from itertools import compress
import argparse
import sys
import numpy as np
import os
import subprocess
import re 

from subprocess import PIPE
from collections import defaultdict

from BayesPaths.UnitigGraph import UnitigGraph
from BayesPaths.UtilsFunctions import convertNodeToName
from BayesPaths.UtilsFunctions import expNormLogProb
from BayesPaths.UtilsFunctions import expLogProb
from operator import itemgetter

from BayesPaths.bnmf_vb import bnmf_vb
from BayesPaths.AugmentedBiGraph import AugmentedBiGraph

import uuid
import networkx as nx
import logging

def gaussianNLL_F(x,f,L):

    return 0.5*(x - f*L)**2

def gaussianNLL_D(x,f,L):

    return -(x - f*L)*L





BETA = 0.6
TAU  = 0.5
MAX_INT_FLOW = 1e6
MAX_REV_FLOW = 1e5
     
     
class ResidualBiGraph():
    """Creates unitig graph"""

    INT_SCALE = 1e5

    def __init__(self, diGraph, sEdges):
        """Empty AugmentedBiGraph"""
        self.diGraph = diGraph
        
        self.sEdges = sEdges
        
        self.rGraph = ResidualBiGraph.createResidualGraph(diGraph):

     
    @classmethod
    def createFromUnitigGraph(cls,unitigGraph):

        assert hasattr(unitigGraph, 'directedUnitigBiGraphS')
    
    
        tempDiGraph = unitigGraph.directedUnitigBiGraphS
    
        copyDiGraph = tempDiGraph.copy()
        
        for node in tempDiGraph.nodes():
        
            pred = list(tempDiGraph.predecessors(node))
        
            if len(pred) > 1 and node != 'sink+':
                newNode = node + 's' 

            tempDiGraph.add_node(newNode)
                
            for pnode in pred:
                
                copyDiGraph.add_edge(pnode,newNode,weight=tempDiGraph[pnode][node]['weight'],
                                            covweight=tempDiGraph[pnode][node]['covweight'],capacity=INT_SCALE,flow=0, weight=0.)
                                            
                copyDiGraph.remove_edge(pnode,node)
                
                copyDiGraph.add_edge(newNode,node,capacity=INT_SCALE,flow=0, weight=0.)
                
                sEdges.add((newNode,node))
            
            elif len(pred) == 1 and node != 'sink+':
                sEdges.add((pred[0],node, capacity=INT_SCALE,flow=0., weight=0.))
        
        biGraph = cls(tempDiGraph, sEdges)
        
        return biGraph
    
    
    @classmethod
    def createResidualGraph(cls,diGraph):
    
        copyDiGraph = diGraph.copy()
        
        for (m,n,f) in diGraph.edges.data('flow', default=0):
            
            copyDiGraph[m,n]['capicity'] = copyDiGraph[m,n]['capicity'] - f
            
            copyDiGraph[m,n]['flow'] = 0
            
            copyDiGraph.add_edge(n,m,capacity=f,flow=0, weight=-copyDiGraph[m,n]['weight'])
    
        return copyDiGraph
    
    def updateCosts(self,vCosts,mapIdx):
    
        for sEdge in self.sEdges:
            
            unitigd = edge[1][:-1]
            
            v = mapIdx[unitigd]
            
            self.diGraph[sEdge[0]][sEdge[1]]['weight'] = vCosts[v]
    
    def updateFlows(self,flowDict, epsilon, mapSedge, newPhi, g):
    
        for sEdge in self.sEdges:
            
            fFlow = self.diGraph[sEdge[0]][sEdge[1]]['flow'] 
            
            v = mapSedge[sEdge]
            
            newPhi[v,g] += flowDict[sEdge[0]][sEdge[1]]
            
            self.diGraph[sEdge[0]][sEdge[1]]['flow'] = fFlow + epsilon*flowDict[sEdge[0]][sEdge[1]]*INT_SCALE

    def initialiseFlows(self):

        for e in self.diGraph.edges:
            self.diGraph[e[0]][e[1]]['flow'] = 0.

            
    def addFlowPath(self, path, pflow):

        for u,v in zip(path,path[1:]):
            self.diGraph[u][v]['flow'] += pflow

    def addEdgePath(self, path, pflow):
    
        for e in path:
            fC = self.diGraph[e[0]][e[1]]['flow']
            
            fN = max(fC + pflow,0)
        
            self.diGraph[e[0]][e[1]]['flow'] = fN


class NMFGraph():


    DELTA = 1.0e-6
    EPSILON = 1.0e-5

    def __init__(self, unitigGraph, X, G, lengths, mapIdx):

        self.unitigGraph = unitigGraph
        
        self.X = X
        
        (self.V, self.S) = self.X.shape
        
        self.G = G
        
        self.biGraphs = {}
        
        for g in range(self.G):
            
            self.biGraphs[g] = ResidualBiGraph.createFromUnitigGraph(unitigGraph)
            
            self.biGraphs.initialiseFlows()
            

        self.gamma = np.zeros((self.G,self.S))
        
        self.phi = np.zeros((self.V,self.S))
    
        self.mapIdx = mapIdx
    
    def optimiseFlows(self):
    
    
        iter = 0
        
        while iter < 100:
        
            #first compute phi gradient in matrix format
            
            eLambda = (np.dot(self.phi*self.gamma) + self.DELTA) * self.lengths[:,np.newaxis]
            
            gSum = np.sum(self.gamma,axis=0)
            R = self.X/eLambda
            gradPhi = (- R + gSum[np.newaxis,:])*self.lengths[:,np.newaxis]
        
            newPhi = np.copy(self.phi)
            
            for g in range(self.G):
                self.biGraphs[g].updateCosts(gradPhi[:,g],mapIdx) 
            
                residualGraph = ResidualBiGraph.createResidualGraph(self.biGraphs[g])
                
                flowCost, flowDict = nx.network_simplex(residualGraph)
                 
                self.biGraphs[g].updateFlows(flowDict,EPSILON, mapSedge, newPhi, g)
        
        
            pL = self.phi*self.lengths[:.np.newaxis]
            pSum =  np.sum(pL,axis=0)
        
            gradGamma = (- np.dot(np.transpose(pL),R) + pSum[:.np.newaxis])
        
        
            self.gamma = self.gamma + gradGamma*EPSILON
            
            self.gamma[self.gamma < 0] = 0.
                    
            self.phi = newPhi
        
            iter = iter+1
    

    def evalPathWeight(self, path, weight):

        D  = 0.0

        for u,v in zip(path,path[1:]):
            D += self.diGraph[u][v][weight]
    
        return D



    
def readCogStopsDead(cog_graph,kmer_length,cov_file):

    deadEndFile = cog_graph[:-3] + "deadends"
        
    stopFile = cog_graph[:-3] + "stops"

    try:
        unitigGraph = UnitigGraph.loadGraphFromGfaFile(cog_graph,int(kmer_length), cov_file, tsvFile=True, bRemoveSelfLinks = True)
    except IOError:
        print('Trouble using file {}'.format(cog_graph))
        sys.exit()
    
        
    deadEnds = []


    try:
        with open(deadEndFile) as f:
            for line in f:
                line.strip()
                deadEnds.append(line)
    except IOError:
        print('Trouble using file {}'.format(deadEndFile))
        sys.exit()
        
    stops = []
        
    try:
        with open(stopFile) as f:
            for line in f:
                line = line.strip()
                toks = line.split("\t")
                dirn = True
                if toks[1] == '-':
                    dirn = False
                stops.append((toks[0],dirn))
    except IOError:
        print('Trouble using file {}'.format(stopFile))
        sys.exit()
        
    return (unitigGraph, stops, deadEnds )
 
 
#def removeDegenerate(haplotypes,paths):



def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("cog_graph", help="gfa file")
    
    parser.add_argument("kmer_length", help="kmer length assumed overlap")
    
    parser.add_argument("cov_file", help="tsv file")
    
    args = parser.parse_args()

    import ipdb; ipdb.set_trace()    

    np.random.seed(seed=12637)

    (unitigGraph, stops, deadEnds ) = readCogStopsDead(args.cog_graph,args.kmer_length,args.cov_file)
    
    
    (source_list, sink_list) = unitigGraph.selectSourceSinksStops(stops, deadEnds, 3000)

    source_names = [convertNodeToName(source) for source in source_list] 

    sink_names = [convertNodeToName(sink) for sink in sink_list]
    
    unitigGraph.setDirectedBiGraphSource(source_names,sink_names)
    
    nx.write_adjlist(unitigGraph.directedUnitigBiGraphS,"diGraphS.adjlist")
    
    
    adjLengths = {}
    
    covMapAdj = {}
    
    readLength = 150.
    #readLength = 1
    kFactor = readLength/(readLength - unitigGraph.overlapLength + 1.)
    
    kFactor = 1.
    for unitig in unitigGraph.unitigs:
 
        adjLengths[unitig] =  unitigGraph.lengths[unitig] - 2.0*unitigGraph.overlapLength + readLength
        
        #adjLengths[unitig] = 1.
        covMapAdj[unitig] = unitigGraph.covMap[unitig] * float(adjLengths[unitig])*(kFactor/readLength)
        
        
    V = len(unitigGraph.unitigs)
    S = unitigGraph.covMap[unitigGraph.unitigs[0]].shape[0]
    xValsU = {}
    X = np.zeros((V,S))
    M = np.ones((V,S))
    v = 0
    mapUnitigs = {}
    with open('coverage.csv','w') as f:            
        for unitig in unitigGraph.unitigs:
            readSum = np.sum(covMapAdj[unitig])
            mapUnitigs[unitig] = v
            xValsU[unitig] = readSum 
            covSum = np.sum(unitigGraph.covMap[unitig])*kFactor
            X[v,:] = unitigGraph.covMap[unitig]/adjLengths[unitig]
            f.write(unitig + ',' + str(unitigGraph.lengths[unitig]) +',' + str(covSum) + ',' + str(readSum) + '\n') 
            v+=1
    
    
    hyperp = { 'alphatau':0.1, 'betatau':0.1, 'alpha0':1.0e-6, 'beta0':1.0e-6, 'lambdaU':1.0e-3, 'lambdaV':1.0e-3}
        
    BNMF = bnmf_vb(X,M,10,True,hyperparameters=hyperp)
        
    BNMF.initialise()      
        
    BNMF.run(1000)
    
    unitigGraph.setDirectedBiGraphSource(source_names,sink_names)
    
    xVals = {}
    Lengths = {}
    
    augmentedBiGraph = AugmentedBiGraph.createFromUnitigGraph(unitigGraph)
    
    for edge in augmentedBiGraph.sEdges:
        unitigd = edge[1][:-1]
        v = mapUnitigs[unitigd]
        xVals[edge] = BNMF.exp_U[v,0]
        #xVals[edge] = xValsU[unitigd]
        #Lengths[edge] = adjLengths[unitigd]
        Lengths[edge] = 1.
    
    augmentedBiGraph.X = xVals
    augmentedBiGraph.L = Lengths
    
    augmentedBiGraph.optimseFlows(gaussianNLL_F, gaussianNLL_D, 1000)

    augmentedBiGraph.decomposeFlows()
    
    nmfGraph = NMFGraph(unitigGraph, xVals, 10, adjLengths, mapUnitigs)
        
    #nx.write_graphml(unitigGraph.augmentedUnitigBiGraphS,"test.graphml")

    #def dijkstra_path(G, source, target, weight='weight')

if __name__ == "__main__":
    main(sys.argv[1:])
