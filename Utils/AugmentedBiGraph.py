from itertools import compress
import argparse
import sys
import numpy as np
import os
import subprocess
import re 

from subprocess import PIPE
from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from collections import defaultdict

from Utils.UnitigGraph import UnitigGraph
from Utils.UtilsFunctions import convertNodeToName
from Utils.UtilsFunctions import expNormLogProb
from Utils.UtilsFunctions import expLogProb
from kmedoids.kmedoids import kMedoids
from operator import itemgetter

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
        

class AugmentedBiGraph():
    """Creates unitig graph"""


    def __init__(self, diGraph, sEdges):
        """Empty AugmentedBiGraph"""
        self.diGraph = diGraph
        
        self.sEdges = sEdges
        
        self.rGraph = self.diGraph.reverse(copy=True)
        
    @classmethod
    def createFromUnitigGraph(cls,unitigGraph):

        assert hasattr(unitigGraph, 'directedUnitigBiGraphS')
        
        tempDiGraph = AugmentedBiGraph.removeCycles(unitigGraph.directedUnitigBiGraphS)
        
        copyDiGraph = tempDiGraph.copy()
        
        top_sort = list(nx.topological_sort(tempDiGraph))
        
        sEdges = set()
        
        for node in top_sort:
            
            pred = list(copyDiGraph.predecessors(node))
            
            if len(pred) > 1 and node != 'sink+':
                newNode = node + 's' 
            
                tempDiGraph.add_node(newNode)
                
                for pnode in pred:
                
                    tempDiGraph.add_edge(pnode,newNode,weight=copyDiGraph[pnode][node]['weight'],
                                            covweight=copyDiGraph[pnode][node]['covweight'])
                                            
                    tempDiGraph.remove_edge(pnode,node)
                
                tempDiGraph.add_edge(newNode,node)
                sEdges.add((newNode,node))
            
            elif len(pred) == 1 and node != 'sink+':
                sEdges.add((pred[0],node))
        
        biGraph = cls(tempDiGraph, sEdges)
        
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
                
                cGraph.add_edge(lastSink,gene + '_source+', weight=0.,covweight=0.)
            
            lastGene = gene
        
        biGraph = cls(cGraph, sEdges)
        
        return biGraph
        
        
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
            

    def evalPathWeight(self, path, weight):

        D  = 0.0

        for u,v in zip(path,path[1:]):
            D += self.diGraph[u][v][weight]
    
        return D


    def evalDF(self, fF, derivF):

        D = 0.
        F = 0.
    
        for e in self.sEdges:
            D += derivF(self.X[e],self.diGraph[e[0]][e[1]]['flow'],self.L[e])
            F += fF(self.X[e],self.diGraph[e[0]][e[1]]['flow'],self.L[e])
    
        return (D, F)


    def setWeightsD(self, derivF):
    
        for e in self.diGraph.edges:
        
            dVal = 0.
    
            if e in self.sEdges:
                dVal = derivF(self.X[e],self.diGraph[e[0]][e[1]]['flow'],self.L[e])
        
            self.diGraph[e[0]][e[1]]['dweight'] = dVal

    def setResidualGraph(self):
    
        fMax = max(dict(self.diGraph.edges).items(), key=lambda x: x[1]['flow'])[1]['flow']
        
        wMax = max(dict(self.diGraph.edges).items(), key=lambda x: x[1]['dweight'])[1]['dweight']
        
        wMin = min(dict(self.diGraph.edges).items(), key=lambda x: x[1]['dweight'])[1]['dweight']
        
        wRange = wMax - wMin 
    
        for e in self.diGraph.edges:
        
            if fMax > 0:    
                self.rGraph[e[1]][e[0]]['capacity'] = int((self.diGraph[e[0]][e[1]]['flow']/fMax)*MAX_INT_FLOW)  
            
    
            if e in self.sEdges and wRange > 0:
                    self.rGraph[e[1]][e[0]]['rweight'] = int((-self.diGraph[e[0]][e[1]]['dweight']/wRange)*MAX_INT_FLOW)
    
        return (fMax,wRange)


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
    
    
    def deltaF(self, spath, pflow, NLL_F):
    
        DeltaF = 0.       
        
        for es in spath:
            fC = self.diGraph[es[0]][es[1]]['flow']
            
            fN = max(fC + pflow,0)
            
            DeltaF += NLL_F(self.X[es],fN,self.L[es]) - NLL_F(self.X[es],fC,self.L[es])
                

        return DeltaF
        
    def optimseFlows(self, logger,  NLL_F, NLL_D, maxIter):
    
        self.initialiseFlows()  
   
        dF = 0.
        F = 0.
    
        (dF, F) = self.evalDF(NLL_F, NLL_D)
    
        rho = 1.0e2
    
        i = 0
    
        ssedges = set(self.sEdges) 
    
        init_pflow = 0.1*max(self.X.items(), key=itemgetter(1))[1]
        logger.info("Performing %d iterations of graph normalisation: ",maxIter)
        logger.info("Iter, dF, F")
        
        while i < maxIter:
   
            self.setWeightsD(NLL_D)
            
            path = nx.bellman_ford_path(self.diGraph, 'source+', 'sink+', weight='dweight')
    
            weight = self.evalPathWeight(path, 'dweight')

            epath = [(u,v) for u,v in zip(path,path[1:])]
        
            spath = set(epath) & ssedges
            
            if i > 10 and i % 5 == 0: 
        
                (fMax, rMax) = self.setResidualGraph()
                
                if fMax > 1.0e-6 and rMax > 1.0e-6:
                    ds = {'sink+': -MAX_REV_FLOW,  'source+': MAX_REV_FLOW}
                
                    nx.set_node_attributes(self.rGraph, ds, 'demand')
                
                    (mf,pf) = nx.network_simplex(self.rGraph, demand='demand', capacity='capacity', weight='rweight')
            
                    fCost = (mf*rMax)/(MAX_INT_FLOW*MAX_REV_FLOW)
                else:
                    fCost = 1.0e6
            else:
                fCost = 1.0e6
                
            if weight < fCost and weight < 0.:
                pflow = init_pflow 
            
                DeltaF = self.deltaF(spath, pflow, NLL_F)
            
                while DeltaF > pflow*weight*BETA:
                    pflow *= TAU
                
                    DeltaF = self.deltaF(spath, pflow, NLL_F)

                if pflow > 0.:
                    self.addFlowPath(path, pflow)
                
            else:
                if fCost < 0.:
                    epath = []
                    
                    for k, v in pf.items():
                        for k2, v2 in v.items():
                            if int(v2) > 0:
                                epath.append((k2,k))
                
                    pflow = fMax*(MAX_REV_FLOW/MAX_INT_FLOW) 
                
                    spath = set(epath) & ssedges 
                
                    DeltaF = self.deltaF(spath, -pflow, NLL_F)
                    
                    while DeltaF > pflow*fCost*BETA:
                        pflow *= TAU
                
                        DeltaF = self.deltaF(spath, -pflow, NLL_F)
                    
                        #print(str(pflow) + ',' + str(DeltaF))
 
                    if pflow > 0.:
                        self.addEdgePath(epath, -pflow)
         

            (dF, F) = self.evalDF(NLL_F, NLL_D)
        
            if i % 10 == 0:
                logger.info("%d, %f, %f", i, dF, F)
            

            i+=1

    def decomposeFlows(self):
      
        paths = {}
      
        maxFlow = 1.0
        
        while maxFlow > 0.:
    
            (maxPath, maxFlow) = self.getMaxMinFlowPathDAG()
        
            self.addFlowPath(maxPath, -maxFlow)
        
            paths[tuple(maxPath)] = maxFlow
        
            print(str(maxFlow))
    
        return paths


    
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
        
    #nx.write_graphml(unitigGraph.augmentedUnitigBiGraphS,"test.graphml")

    #def dijkstra_path(G, source, target, weight='weight')

if __name__ == "__main__":
    main(sys.argv[1:])
