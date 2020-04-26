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

#from abc import ABC, abstractmethod

#class Convex1DFunction(ABC):
 #   @abstractmethod
  #  def F(self,x):
   #     pass
    
    #def D(self,x):
     #   pass

def gaussianNLL_F(x,f,L):

    return 0.5*(x - f*L)**2

def gaussianNLL_D(x,f,L):

    return -(x - f*L)*L
    

def initialiseFlows(graph):

    for e in graph.edges:
        graph[e[0]][e[1]]['flow'] = 0.

def addFlowPath(graph, path, pflow):

    for u,v in zip(path,path[1:]):
        graph[u][v]['flow'] += pflow

def evalPathWeight(graph, path, weight):

    D  = 0.0

    for u,v in zip(path,path[1:]):
        D += graph[u][v][weight]
    
    return D


def evalDF(graph, fF, derivF, sedges, xVals, Lengths):

    D = 0.
    F = 0.
    
    for e in sedges:
        D += derivF(xVals[e],graph[e[0]][e[1]]['flow'],Lengths[e])
        F += fF(xVals[e],graph[e[0]][e[1]]['flow'],Lengths[e])
    
    return (D, F)


def setWeightsD(graph, derivF, sedges, xVals, Lengths):
    
    for e in graph.edges:
        
        dVal = 0.
    
        if e in sedges:
            dVal = derivF(xVals[e],graph[e[0]][e[1]]['flow'],Lengths[e])
        
        graph[e[0]][e[1]]['dweight'] = dVal
        graph[e[0]][e[1]]['rweight'] = -dVal

def getMaxMinFlowPathDAG(dGraph):
    
    assert nx.is_directed_acyclic_graph(dGraph)
    
    top_sort = list(nx.topological_sort(dGraph))
    lenSort = len(top_sort)
            
    maxPred = {}
    maxFlowNode = {}
    
    for node in top_sort:
        pred = list(dGraph.predecessors(node))
            
        if len(pred) > 0:
            maxFlowPred = min(maxFlowNode[pred[0]],dGraph[pred[0]][node]['flow'])
            maxPred[node] = pred[0]
                
            for predecessor in pred[1:]:
            #    print (node + "," + predecessor + "," + str(dGraph[predecessor][node]['flow']))
                
                weight =  min(maxFlowNode[predecessor],dGraph[predecessor][node]['flow'])
                
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
        
        
    
    xValsU = {}
    
    with open('coverage.csv','w') as f:            
        for unitig in unitigGraph.unitigs:
            readSum = np.sum(covMapAdj[unitig])
            xValsU[unitig] = readSum 
            covSum = np.sum(unitigGraph.covMap[unitig])*kFactor
            
            f.write(unitig + ',' + str(unitigGraph.lengths[unitig]) +',' + str(covSum) + ',' + str(readSum) + '\n') 

    augmentedBiGraph = unitigGraph.getAugmentedBiGraphSource(source_names,sink_names)
    
    sedges = unitigGraph.sEdges
    
    xVals = {}
    Lengths = {}
    
    for edge in sedges:
        unitigd = edge[1][:-1]
        xVals[edge] = xValsU[unitigd]
        Lengths[edge] = adjLengths[unitigd]
        
    initialiseFlows(augmentedBiGraph)  
   
    dF = 0.
    F = 0.
    
    (dF, F) = evalDF(augmentedBiGraph, gaussianNLL_F, gaussianNLL_D, sedges, xVals, Lengths)
    
    rho = 1.0e2
    
    i = 0
    
    Beta = 0.6
    Tau = 0.5
    
    ssedges = set(sedges) 
    
    while i < 1000:
   
        setWeightsD(augmentedBiGraph, gaussianNLL_D, sedges, xVals, Lengths)
    
        path = nx.bellman_ford_path(augmentedBiGraph, 'source+', 'sink+', weight='dweight')
    
        weight = evalPathWeight(augmentedBiGraph, path, 'dweight')

        epath = [(u,v) for u,v in zip(path,path[1:])]
        
        spath = set(epath) & ssedges 
        
        if weight < 0.0:
            pflow = 0.1 
            
            
            DeltaF = 0.
            for es in spath:
                fC = augmentedBiGraph[es[0]][es[1]]['flow']
                DeltaF += gaussianNLL_F(xVals[es],fC + pflow,Lengths[es]) - gaussianNLL_F(xVals[es],fC,Lengths[es])
            
            while DeltaF > pflow*weight*Beta:
                pflow *= Tau
                
                DeltaF = 0.       
                for es in spath:
                    fC = augmentedBiGraph[es[0]][es[1]]['flow']
                    DeltaF += gaussianNLL_F(xVals[es],fC + pflow,Lengths[es]) - gaussianNLL_F(xVals[es],fC,Lengths[es])


            if pflow > 0.:
                addFlowPath(augmentedBiGraph, path, pflow)
        else:
            print("Debug")
        
        (dF, F) = evalDF(augmentedBiGraph, gaussianNLL_F, gaussianNLL_D, sedges, xVals, Lengths)

        print(str(i) + "," + str(dF) + "," + str(F))

        i+=1

    import ipdb; ipdb.set_trace()


    maxFlow = 1.0
    while maxFlow > 0.:
    
        (maxPath, maxFlow) = getMaxMinFlowPathDAG(augmentedBiGraph)
        
        addFlowPath(augmentedBiGraph, maxPath, -maxFlow)
        
        print(str(maxFlow))
        
    #nx.write_graphml(unitigGraph.augmentedUnitigBiGraphS,"test.graphml")

    #def dijkstra_path(G, source, target, weight='weight')

if __name__ == "__main__":
    main(sys.argv[1:])
