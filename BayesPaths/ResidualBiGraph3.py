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

import uuid
import networkx as nx
import logging


def lassoF(phi):

    if phi < 0.5:
        return phi
    else:
        return 1 - phi

def lassoPenalty(phiMatrix):

    fL = 0.
    for i in range(phiMatrix.shape[0]):
        for j in range(phiMatrix.shape[1]):
            fL += lassoF(phiMatrix[i][j])

    return fL

INT_SCALE = 1.0e6
BETA = 0.6
TAU  = 0.5

class ResidualBiGraph():
    """Creates unitig graph for minimisation"""


    def __init__(self, diGraph, sEdges, maxFlow = 1.):
        """Empty AugmentedBiGraph"""
        self.diGraph = diGraph
        
        self.sEdges = sEdges

        self.maxFlow = maxFlow
        
        self.rGraph = ResidualBiGraph.createResidualGraph(diGraph)

     
    @classmethod
    def createFromUnitigGraph(cls,unitigGraph, maxFlow = 1.):

        assert hasattr(unitigGraph, 'directedUnitigBiGraphS')
    
        tempDiGraph = ResidualBiGraph.removeCycles(unitigGraph.directedUnitigBiGraphS)
    
        copyDiGraph = tempDiGraph.copy()
        
        sEdges = set()
        
        for node in tempDiGraph.nodes():
        
            pred = list(tempDiGraph.predecessors(node))
        
            if len(pred) > 1 and node != 'sink+':
                newNode = node + 's' 

                copyDiGraph.add_node(newNode)
                
                for pnode in pred:
                
                    copyDiGraph.add_edge(pnode,newNode,weight=tempDiGraph[pnode][node]['weight'], 
                                        covweight=tempDiGraph[pnode][node]['covweight'],capacity=maxFlow*INT_SCALE,flow=0)
                                            
                    copyDiGraph.remove_edge(pnode,node)
                
                    copyDiGraph.add_edge(newNode,node,capacity=maxFlow*INT_SCALE,flow=0, weight=0.)
                
                    sEdges.add((newNode,node))
            
            elif len(pred) == 1 and node != 'sink+':
                copyDiGraph.add_edge(pred[0],node,weight=tempDiGraph[pred[0]][node]['weight'], 
                                        covweight=tempDiGraph[pred[0]][node]['covweight'],capacity=maxFlow*INT_SCALE,flow=0)
                    
                sEdges.add((pred[0],node))
        
        
        nx.set_edge_attributes(copyDiGraph, maxFlow*INT_SCALE, name='capacity')
        nx.set_edge_attributes(copyDiGraph, 0, name='flow')
        nx.set_edge_attributes(copyDiGraph, 0, name='weight')
        
        attrs = {'source+': {'demand': -INT_SCALE}, 'sink+': {'demand': INT_SCALE}}

        nx.set_node_attributes(copyDiGraph, attrs)
        
        biGraph = cls(copyDiGraph, sEdges, maxFlow)
        
        return biGraph
    
    @classmethod
    def removeCycles(cls, inGraph):
        
        diGraph = inGraph.copy()
        
        while not nx.is_directed_acyclic_graph(diGraph):
        
            cycle = nx.find_cycle(diGraph)
            
            weakestLink = sys.float_info.max
            weakestEdge = None
            
            for edge in cycle:
                weight = diGraph[edge[0]][edge[1]]['covweight']
            
                if weight < weakestLink:
                    weakestEdge = edge
                    weakestLink = weight
            
            diGraph.remove_edge(weakestEdge[0],weakestEdge[1])
   
        return diGraph
    
    @classmethod
    def createResidualGraph(cls,diGraph):
    
        copyDiGraph = diGraph.copy()
        
        
        for (m,n,f) in diGraph.edges.data('flow', default=0):
                        
            copyDiGraph[m][n]['capacity'] = max(0,diGraph[m][n]['capacity'] - f)
            
            copyDiGraph[m][n]['flow'] = 0
            
            copyDiGraph.add_edge(n,m,capacity=f,flow=0, weight=-diGraph[m][n]['weight'])
    
    
        nx.set_node_attributes(copyDiGraph,0.0,'demand')
        
        return copyDiGraph
    
    @classmethod
    def combineGraphs(cls,dictBiGraphs,geneList,mapGeneIdx,maxFlow = 1.):
      
        cGraph = nx.DiGraph()
        
        lastGene = None
        
        sEdges = set()
        
        newGeneIdx = {}
        defaultdict(dict)
        
        for gene in geneList:
        
            unitigsDash = list(dictBiGraphs[gene].diGraph.nodes())
            
            mapNodes = {s:gene + "_" + s for s in unitigsDash}
            
            for (ud, mapunitig) mapNodes.items():
                newGeneIdx[mapunitig] = mapGeneIdx[gene][ud]
            
            
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
                
                cGraph.add_edge(lastSink,gene + '_source+', weight=0,covweight=0.,capacity=maxFlow*INT_SCALE,flow=0)
            
            lastGene = gene
        
        biGraph = cls(cGraph, sEdges, maxFlow)
        
        return (biGraph, newGeneIdx)
    
    
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
                

    def deltaF(self, flowDict, epsilon, X, eLambda, mapIdx, Lengths, bKLDivergence = False, bLasso = False, fLambda = 1.):
    
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
                niFlow = int(iFlow - epsilon*flowDict[outnode][node])
                nfFlow =  float(niFlow)/INT_SCALE
                change = True
                
        
            if change:
                newLambda = eLambda[v] + Lengths[v]*(nfFlow - fFlow)
                
                if bKLDivergence:
                    T1 = newLambda - eLambda[v]
                
                    T2 = X[v]*np.log(newLambda/eLambda[v])
                
                    DeltaF += np.sum(T1 - T2)
                else:
                    DeltaF += 0.5*np.sum((X[v] - newLambda)**2 - (X[v] - eLambda[v])**2) 
        
                if bLasso:
                    DeltaF += fLambda*(nfFlow - fFlow)
            
        return DeltaF


    def initialiseFlows(self):

        for e in self.diGraph.edges:
            self.diGraph[e[0]][e[1]]['flow'] = 0

            
    def addFlowPath(self, path, pflow):

        for u,v in zip(path,path[1:]):
            #print(u + ',' + v)
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
        
    def updatePhi(self, phi, mapIdx):
    
        for sEdge in self.sEdges:
        
            iFlow = self.diGraph[sEdge[0]][sEdge[1]]['flow']
        
            fFlow = float(iFlow)/INT_SCALE
        
            #print(str(fFlow))
        
            unitigd = sEdge[1][:-1]
        
            v = mapIdx[unitigd] 
    
            phi[v] = fFlow
    
    def decomposeFlows(self):
      
        paths = {}
      
        maxFlow = 1.0
        
        while maxFlow > 0.:
    
            (maxPath, maxFlow) = self.getMaxMinFlowPathDAG()
        
            self.addFlowPath(maxPath, -maxFlow)
        
            paths[tuple(maxPath)] = maxFlow/INT_SCALE
        
            print(str(maxFlow/INT_SCALE))
    
        return paths
    
    def getMaxMinFlowPathDAG(self):
    
        #self.initialiseFlows()  
    
        self.top_sort = list(nx.topological_sort(self.diGraph))
    
        lenSort = len(self.top_sort)
            
        maxPred = {}
        maxFlowNode = {}
    
        for node in self.top_sort:
            pred = list(self.diGraph.predecessors(node))
            
            if len(pred) > 0:
                maxFlowPred = min(maxFlowNode[pred[0]],self.diGraph[pred[0]][node]['flow'])
                maxPred[node] = pred[0]
                
                for predecessor in pred[1:]:
            #    print (node + "," + predecessor + "," + str(dGraph[predecessor][node]['flow']))
                
                    weight =  min(maxFlowNode[predecessor],self.diGraph[predecessor][node]['flow'])
                
                    if weight > maxFlowPred:
                        maxFlowPred = weight
                        maxPred[node] = predecessor
                
                maxFlowNode[node]  = maxFlowPred
            else:
                maxFlowNode[node] = sys.float_info.max
                maxPred[node] = None
            
        minPath = []
        bestNode = 'sink+'
        while bestNode is not None:
            minPath.append(bestNode)
            bestNode = maxPred[bestNode]
        
        minPath.pop(0)
        minPath.pop()
        minPath.reverse()
                            
        
        return (minPath, maxFlowNode['sink+'])

    

class FlowGraphML():


    DELTA = 1.0e-9
    EPSILON = 1.0e-5
    PRECISION = 1.0e-15

    def __init__(self, biGraphs, genes, prng, X, lengths, mapGeneIdx, mask = None, bLasso = False, fLambda = 1.0):
        
        self.X = X
        
        self.V = self.X.shape[0]
        
        if mask is None:
            self.mask = np.ones((self.V))
        else:
            self.mask = mask
        
        self.Omega = np.sum(self.mask > 0)
        
        
        self.biGraphs = {}
        
        for (gene,biGraph) in biGraphs.items():
            self.biGraphs[gene] = ResidualBiGraph(biGraph.diGraph.copy(),biGraph.sEdges)
            
            self.biGraphs[gene].initialiseFlows()
        
        self.genes = genes
        
        self.mapGeneIdx  = mapGeneIdx
        
        self.phi = np.zeros((self.V))
        
        for gene, biGraph in self.biGraphs.items():
                
            pathg = biGraph.getRandomPath(prng)
    
            biGraph.addFlowPath(pathg, INT_SCALE)
            
            for u in pathg:
                ud = u[:-1]
                
                if ud in self.mapGeneIdx[gene]:
                
                    v = self.mapGeneIdx[gene][ud]
                
                    self.phi[v] = 1.
      
        self.tau = 1. 
    
        self.lengths = lengths
       
        self.bLasso = bLasso

        self.fLambda = fLambda        
 
    
    def _KDivergence(self, eLambda, mask):
        
        return np.sum(mask*(eLambda - self.X*np.log(eLambda)))

    def _FDivergence(self, eLambda, mask):
        
        return 0.5*np.sum(np.square(mask*(eLambda - self.X)))
    
    def optimiseFlows(self, max_iter=500, bKLDivergence = False):
    
        iter = 0
        
        eLambda = (self.phi + self.DELTA) * self.lengths
        
        if bKLDivergence:
            NLL1 = self._KDivergence(eLambda, self.mask)
        else:
            NLL1 = self._FDivergence(eLambda, self.mask)

        print(str(iter) + "," + str(NLL1))
        
        while iter < max_iter:
        
            #first compute phi gradient in matrix format
            
            eLambda = (self.phi + self.DELTA) * self.lengths
        
            R = self.X/eLambda
            
            if bKLDivergence:
                gradPhi = - R*self.mask + self.lengths
            else:
    
                gradPhi = (eLambda - self.X)*self.mask*self.lengths
                
            
            if self.bLasso:
                gradPhi +=  self.fLambda
                
        
            newPhi = np.copy(self.phi)
            
            
            for gene, biGraph in self.biGraphs.items():
                    
                biGraph.updateCosts(gradPhi,self.mapGeneIdx[gene]) 
            
                residualGraph = ResidualBiGraph.createResidualGraph(biGraph.diGraph)
                
                flowCost, flowDict = nx.network_simplex(residualGraph)
                 
                pflow = 0.1 
            
                DeltaF = biGraph.deltaF(flowDict, pflow, self.X, eLambda, self.mapGeneIdx[gene], self.lengths, bKLDivergence, self.bLasso, self.fLambda)               
                weight = flowCost/float(INT_SCALE)
            
                i = 0
                while DeltaF > pflow*weight*BETA and i < 10:
                    pflow *= TAU
                
                    DeltaF = biGraph.deltaF(flowDict, pflow, self.X, eLambda, self.mapGeneIdx[gene], self.lengths, bKLDivergence, self.bLasso, self.fLambda)
        
                    i += 1

                if pflow > 0. and i < 10:                 
                    biGraph.updateFlows(flowDict,pflow)
                
                    
                biGraph.updatePhi(newPhi,self.mapGeneIdx[gene])
         
            
            eLambda1 = (newPhi + self.DELTA) * self.lengths
            
            if bKLDivergence:
                NLL1 = self._KDivergence(eLambda1,self.mask)
            else:
                NLL1 = self._FDivergence(eLambda1,self.mask)
            
            if iter % 1 == 0:        
                print(str(iter) + "," + str(NLL1))
        
                  #print(str(iter) + "," + str(NLL3))
                    
            self.phi = newPhi
        
            iter = iter+1
    
    
    def decomposeFlows(self):

        flowPaths = {}
        for gene, biGraph in self.biGraphs.items():
            flowPaths[gene] = biGraph.decomposeFlows()

        return flowPaths

    def KLDivergence(self,mask):
        
        eLambda = (self.phi + self.DELTA) * self.lengths
        
        div = np.sum(mask*(eLambda - self.X - self.X*np.log(eLambda) + self.X*np.log(self.X + self.PRECISION)))
        
        return div
    
    def FDivergence(self, mask):
        
        eLambda = (self.phi + self.DELTA) * self.lengths
    
        omega = np.sum(mask)

        return np.sqrt(np.sum(np.square(mask*(eLambda - self.X)))/omega)
    

    def evalPathWeight(self, path, weight):

        D  = 0.0

        for u,v in zip(path,path[1:]):
            D += self.diGraph[u][v][weight]
    
        return D

def adjustCoverages(unitigGraph):

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
 
    return (V,S,lengths, mapUnitigs, X)


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
      
    prng = RandomState(args.random_seed) #create prng from seed 

    
    (V, S, lengths, mapUnitigs, X) = adjustCoverages(unitigGraph)


    mapGeneIdx = {}
    mapGeneIdx['gene'] = mapUnitigs 
  
    genes = ['gene']

    residualBiGraphs = {}
    residualBiGraphs['gene'] = ResidualBiGraph.createFromUnitigGraph(unitigGraph)
    
    M = np.ones((V))
    XT = np.sum(X,axis=1)

    flowGraph = FlowGraphML(residualBiGraphs, genes, prng, XT, lengths, mapGeneIdx, M, True, 1.0)    
    flowGraph.bLasso = True        
    flowGraph.fLambda = 1.0e3
    flowGraph.optimiseFlows(50,bKLDivergence = False)

    eLambda =  (flowGraph.phi + flowGraph.DELTA) * flowGraph.lengths
    for v in range(flowGraph.V):
        print(str(v) + ',' + str(flowGraph.X[v]) + ',' +  str(flowGraph.phi[v]) + ',' + str(eLambda[v]))

    paths = flowGraph.decomposeFlows()

    print('Debug')

if __name__ == "__main__":
    main(sys.argv[1:])




