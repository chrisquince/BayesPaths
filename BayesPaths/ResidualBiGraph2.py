from itertools import compress
import argparse
import sys
import numpy as np
import os
import subprocess
import re 

from numpy.random import RandomState
import logging 

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


formatter = logging.Formatter('%(asctime)s %(message)s')


BETA = 0.6
TAU  = 0.5
MAX_INT_FLOW = 1e6
MAX_REV_FLOW = 1e5
INT_SCALE = 1e5
     
class ResidualBiGraph():
    """Creates unitig graph"""

    

    def __init__(self, diGraph, sEdges):
        """Empty AugmentedBiGraph"""
        self.diGraph = diGraph
        
        self.sEdges = sEdges
        
        self.rGraph = ResidualBiGraph.createResidualGraph(diGraph)

     
    @classmethod
    def createFromUnitigGraph(cls,unitigGraph):

        assert hasattr(unitigGraph, 'directedUnitigBiGraphS')
    
    
        tempDiGraph = unitigGraph.directedUnitigBiGraphS
    
        copyDiGraph = tempDiGraph.copy()
        
        sEdges = set()
        
        for node in tempDiGraph.nodes():
        
            pred = list(tempDiGraph.predecessors(node))
        
            if len(pred) > 1 and node != 'sink+':
                newNode = node + 's' 

                copyDiGraph.add_node(newNode)
                
                for pnode in pred:
                
                    copyDiGraph.add_edge(pnode,newNode,weight=tempDiGraph[pnode][node]['weight'], 
                                        covweight=tempDiGraph[pnode][node]['covweight'],capacity=INT_SCALE,flow=0)
                                            
                    copyDiGraph.remove_edge(pnode,node)
                
                    copyDiGraph.add_edge(newNode,node,capacity=INT_SCALE,flow=0, weight=0.)
                
                    sEdges.add((newNode,node))
            
            elif len(pred) == 1 and node != 'sink+':
                copyDiGraph.add_edge(pred[0],node,weight=tempDiGraph[pred[0]][node]['weight'], 
                                        covweight=tempDiGraph[pred[0]][node]['covweight'],capacity=INT_SCALE,flow=0)
                    
                sEdges.add((pred[0],node))
        
        
        nx.set_edge_attributes(copyDiGraph, INT_SCALE, name='capacity')
        nx.set_edge_attributes(copyDiGraph, 0, name='flow')
        nx.set_edge_attributes(copyDiGraph, 0, name='weight')
        
        attrs = {'source+': {'demand': -INT_SCALE}, 'sink+': {'demand': INT_SCALE}}

        nx.set_node_attributes(copyDiGraph, attrs)
        
        biGraph = cls(copyDiGraph, sEdges)
        
        return biGraph
    
    
    @classmethod
    def createResidualGraph(cls,diGraph):
    
        copyDiGraph = diGraph.copy()
        
        for (m,n,f) in diGraph.edges.data('flow', default=0):
                        
            copyDiGraph[m][n]['capacity'] = copyDiGraph[m][n]['capacity'] - f
            
            copyDiGraph[m][n]['flow'] = 0
            
            copyDiGraph.add_edge(n,m,capacity=f,flow=0, weight=-copyDiGraph[m][n]['weight'])
    
    
        nx.set_node_attributes(copyDiGraph,0.0,'demand')
        
        return copyDiGraph
    
    @classmethod
    def combineGraphs(cls,dictBiGraphs,geneList):
      
        cGraph = nx.DiGraph()
        
        lastGene = None
        
        sEdges = set()
        
        for gene in geneList:
        
            unitigsDash = list(dictBiGraphs[gene].diGraph.nodes())
            
            mapNodes = {s:gene + "_" + s for s in unitigsDash}
            
            if lastGene is None:
                mapNodes['source+'] = 'source+'
            
            if gene == geneList[-1]:
                mapNodes['sink+'] = 'sink+'
            
            sMap = [(mapNodes[e[0]],mapNodes[e[1]]) for e in dictBiGraphs[gene].sEdges]
            sEdges.update(sMap)
            
            
            tempGraph = nx.relabel_nodes(dictBiGraphs[gene].diGraph, mapNodes)
 
            cGraph = nx.algorithms.operators.binary.compose(cGraph, tempGraph)
            
            if lastGene is not None:
                lastSink = lastGene + '_sink+'
                
                cGraph.add_edge(lastSink,gene + '_source+', weight=0,covweight=0.,capicity=INT_SCALE,flow=0)
            
            lastGene = gene
        
        biGraph = cls(cGraph, sEdges)
        
        return biGraph
    
    
    def updateCosts(self,vCosts,mapIdx):
    
        for sEdge in self.sEdges:
            
            unitigd = sEdge[1][:-1]
            
            v = mapIdx[unitigd]
            
            self.diGraph[sEdge[0]][sEdge[1]]['weight'] = vCosts[v]/INT_SCALE
    
    def updateFlows(self,flowDict, epsilon):
    
        for (node, flows) in flowDict.items():
        
            for (outnode, flow) in flows.items():
            
                if flow > 0.:
        
                    if self.diGraph.has_edge(node,outnode):
                        fFlow = self.diGraph[node][outnode]['flow'] 
                    
                        self.diGraph[node][outnode]['flow'] = int(fFlow + epsilon*flow)
                
                    else:
                
                        assert self.diGraph.has_edge(outnode,node)
                    
                        fFlow = self.diGraph[outnode][node]['flow'] 
                    
                        self.diGraph[outnode][node]['flow'] = max(0,int(fFlow - epsilon*flow))
                

    def deltaF(self, flowDict, epsilon, X, eLambda, mapIdx, Lengths, g, gamma):
    
        DeltaF = 0.       
        
        for (node,outnode) in self.sEdges:
            nfFlow = 0.
            fFlow = 0.
            v = mapIdx[outnode[:-1]]
            change = False
            iFlow = self.diGraph[node][outnode]['flow']
            fFlow = float(iFlow)/INT_SCALE
            
            if flowDict[node][outnode] > 0.:
                niFlow = int(iFlow + epsilon*flowDict[node][outnode])
                nfFlow =  float(niFlow)/INT_SCALE
                change = True
                    
            elif flowDict[outnode][node] > 0.:                        
                niFlow = int(iFlow - epsilon*flowDict[node][outnode])
                nfFlow =  float(niFlow)/INT_SCALE
                change = True
                
        
            if change:
                newLambda = eLambda[v,:] + Lengths[v]*(nfFlow - fFlow)*gamma[g,:]
                
                T1 = newLambda - eLambda[v,:]
                
                T2 = X[v,:]*np.log(newLambda/eLambda[v,:])
                
                DeltaF += np.sum(T1 - T2)
        
        
        return DeltaF


    def initialiseFlows(self):

        for e in self.diGraph.edges:
            self.diGraph[e[0]][e[1]]['flow'] = 0

            
    def addFlowPath(self, path, pflow):

        for u,v in zip(path,path[1:]):
            self.diGraph[u][v]['flow'] += pflow

    def addEdgePath(self, path, pflow):
    
        for e in path:
            fC = self.diGraph[e[0]][e[1]]['flow']
            
            fN = max(fC + pflow,0)
        
            self.diGraph[e[0]][e[1]]['flow'] = fN

    def getRandomPath(self, prng):
    
        node = 'source+'
        
        path = []
        while node != 'sink+':
            succ = list(self.diGraph.successors(node))
            path.append(node)
            node = prng.choice(succ)
        path.append(node)
        return path
        
    def updatePhi(self, phi,g, mapIdx):
    
        for sEdge in self.sEdges:
        
            iFlow = self.diGraph[sEdge[0]][sEdge[1]]['flow']
        
            fFlow = float(iFlow)/INT_SCALE
        
            #print(str(fFlow))
        
            unitigd = sEdge[1][:-1]
        
            v = mapIdx[unitigd] 
    
            phi[v,g] = fFlow
    

class NMFGraph():


    DELTA = 1.0e-6
    EPSILON = 1.0e-5
    PRECISION = 1.0e-15

    def __init__(self, biGraph, prng, X, G, lengths, mapIdx, bARD = True,alphaG=1.0e-6,betaG=1.0e-6):
        
        self.X = X
        
        (self.V, self.S) = self.X.shape
        
        self.G = G
        
        self.biGraphs = {}
        
        for g in range(self.G):
            
            self.biGraphs[g] = ResidualBiGraph(biGraph.dGraph.copy(),biGraph.sEdges)
            
            self.biGraphs[g].initialiseFlows()
            
        
        scale = 1.0 
        self.gamma = prng.exponential(scale=scale,size=(self.G,self.S))   
        
        self.phi = np.zeros((self.V,self.G))
    
        for g in range(self.G):
            pathg = self.biGraphs[g].getRandomPath(prng)
    
            self.biGraphs[g].addFlowPath(pathg, INT_SCALE)
            
            for u in pathg:
                ud = u[:-1]
                
                if ud in mapIdx:
                
                    v = mapIdx[ud]
                
                    self.phi[v,g] = 1.
                
    
        self.mapIdx  = mapIdx
        self.lengths = lengths
        
        self.bARD = bARD
        if self.bARD:
            self.alphaG = alphaG
            
            self.betaG = betaG
        
            self.alpha_g = np.zeros(self.G)
            
            self.beta_g = np.zeros(self.G)
        
            self.lambda_g = prng.exponential(scale=scale,size=(self.G))  
    
    def optimiseFlows(self, alpha = 1.):
    
    
        iter = 0
        
        eLambda = (np.dot(self.phi,self.gamma) + self.DELTA) * self.lengths[:,np.newaxis]
        NLL1 = np.sum(eLambda - self.X*np.log(eLambda))
        
        print(str(iter) + "," + str(NLL1))
        
        while iter < 200:
        
            #first compute phi gradient in matrix format
            
            eLambda = (np.dot(self.phi,self.gamma) + self.DELTA) * self.lengths[:,np.newaxis]
            gSum = np.sum(self.gamma,axis=1)
            R = self.X/eLambda
            
            gradPhi = (- np.dot(R,self.gamma.transpose()) + gSum[np.newaxis,:])*self.lengths[:,np.newaxis]
        
            gradPhi += -(alpha - 1.)/(self.phi + self.PRECISION) + (alpha - 1.)/(1.0 - self.phi + self.PRECISION)
        
            newPhi = np.copy(self.phi)
            
            for g in range(self.G):
                self.biGraphs[g].updateCosts(gradPhi[:,g],self.mapIdx) 
            
                residualGraph = ResidualBiGraph.createResidualGraph(self.biGraphs[g].diGraph)
                
                flowCost, flowDict = nx.network_simplex(residualGraph)
                 
                pflow = 0.1 
            
                DeltaF = self.biGraphs[g].deltaF(flowDict, pflow, self.X, eLambda, self.mapIdx, self.lengths, g, self.gamma)
                
                weight = flowCost/float(INT_SCALE)
            
                i = 0
                while DeltaF > pflow*weight*BETA and i < 10:
                    pflow *= TAU
                
                    DeltaF = self.biGraphs[g].deltaF(flowDict, pflow, self.X, eLambda, self.mapIdx, self.lengths, g, self.gamma)
        
                    #print(str(i) + "," + str(pflow) + "," + str(DeltaF) + "," + str(pflow*weight*BETA))
                    i += 1

                if pflow > 0. and i < 10:                 
                    self.biGraphs[g].updateFlows(flowDict,pflow)
                
                self.biGraphs[g].updatePhi(newPhi,g,self.mapIdx)
         
            
            eLambda1 = (np.dot(newPhi,self.gamma) + self.DELTA) * self.lengths[:,np.newaxis]
            NLL1 = np.sum(eLambda1 - self.X*np.log(eLambda1))
            print(str(iter) + "," + str(NLL1))
        
            pL = self.phi*self.lengths[:,np.newaxis]
            pSum =  np.sum(pL,axis=0)
            
            if self.bARD:
                pSum += self.lambda_g
        
            self.gamma = self.gamma*(np.dot(np.transpose(pL),R)/pSum[:,np.newaxis]) 
            
            #self.gamma[self.gamma < 0] = 0.
            
            eLambda3 = (np.dot(newPhi,self.gamma) + self.DELTA) * self.lengths[:,np.newaxis]
            NLL3 = np.sum(eLambda3 - self.X*np.log(eLambda3))
            
            
            if self.bARD:
                self.alpha_g = self.alphaG + self.S
            
                self.beta_g = self.betaG + np.sum(self.gamma,axis=1)
                
                self.lambda_g = self.alpha_g/self.beta_g
            
            print(str(iter) + "," + str(NLL3))
                    
            self.phi = newPhi
        
            iter = iter+1
    

    def evalPathWeight(self, path, weight):

        D  = 0.0

        for u,v in zip(path,path[1:]):
            D += self.diGraph[u][v][weight]
    
        return D


    def getMaxPaths(self):
    
        maxPaths = {}
    
        for g in range(self.G):
            gGraph = self.biGraphs[g].diGraph
            for (u, v, c) in gGraph.edges.data('flow', default=0.0):
            
                fFlow = float(c)/INT_SCALE
                
                gGraph[u][v]['weight'] = -np.log(fFlow + 1.0e-12)
                
            
            path = nx.shortest_path(self.biGraphs[g].diGraph,'source+','sink+', weight='weight', method='dijkstra')
            
            
            path.pop(0)
            path.pop()
            pathN = []
            for p in path:
                if p[-1] != 's':
                    pathN.append(p)
            
            maxPaths[g] = pathN

        return maxPaths
    
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
    
    parser.add_argument('-s', '--random_seed', default=23724839, type=int,
        help="specifies seed for numpy random number generator defaults to 23724839 applied after random filtering")
    
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
    
    for unitig in unitigGraph.unitigs:
 
        adjLengths[unitig] =  unitigGraph.lengths[unitig] - 2.0*unitigGraph.overlapLength + readLength
        
        #adjLengths[unitig] = 1.
        covMapAdj[unitig] = unitigGraph.covMap[unitig] * float(adjLengths[unitig])*(kFactor/readLength)
        
        
    V = len(unitigGraph.unitigs)
    S = unitigGraph.covMap[unitigGraph.unitigs[0]].shape[0]
    xValsU = {}
    X = np.zeros((V,S))
    lengths = np.zeros(V)
    M = np.ones((V,S))
    v = 0
    mapUnitigs = {}
    with open('coverage.csv','w') as f:            
        for unitig in unitigGraph.unitigs:
            readSum = np.sum(covMapAdj[unitig])
            mapUnitigs[unitig] = v
            xValsU[unitig] = readSum 
            covSum = np.sum(unitigGraph.covMap[unitig])*kFactor
            X[v,:] = unitigGraph.covMap[unitig] * float(adjLengths[unitig])*(kFactor/readLength)
            lengths[v] =  float(adjLengths[unitig])
            f.write(unitig + ',' + str(unitigGraph.lengths[unitig]) +',' + str(covSum) + ',' + str(readSum) + '\n') 
            v+=1
    
    
    hyperp = { 'alphatau':0.1, 'betatau':0.1, 'alpha0':1.0e-6, 'beta0':1.0e-6, 'lambdaU':1.0e-3, 'lambdaV':1.0e-3}
  
    prng = RandomState(args.random_seed) #create prng from seed 

    #set log file
    logFile="test.log"
    
    handler = logging.FileHandler(logFile)        
    handler.setFormatter(formatter)

    mainLogger = logging.getLogger('main')
    mainLogger.setLevel(logging.DEBUG)
    mainLogger.addHandler(handler)
  
    
    nmfGraph = NMFGraph(prng, unitigGraph, X, 16, lengths, mapUnitigs)
    
    nmfGraph.optimiseFlows(alpha=1.)
    
    maxPaths = nmfGraph.getMaxPaths()
    
    for g in range(nmfGraph.G):
        seq = unitigGraph.getUnitigWalk(maxPaths[g])
        
        print(">Hap_" + str(g) + "\n" + seq)
    
    nmfGraph.optimiseFlows(alpha=0.5)

    maxPaths = nmfGraph.getMaxPaths()
    
    for g in range(nmfGraph.G):
        seq = unitigGraph.getUnitigWalk(maxPaths[g])
        
        print(">Hap_" + str(g) + "\n" + seq)

    
    print('Debug')
    #nx.write_graphml(unitigGraph.augmentedUnitigBiGraphS,"test.graphml")

    #def dijkstra_path(G, source, target, weight='weight')

if __name__ == "__main__":
    main(sys.argv[1:])
