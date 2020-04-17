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
    
    kFactor = readLength/(readLength - unitigGraph.overlapLength + 1.)
    
    for unitig in unitigGraph.unitigs:
 
        adjLengths[unitig] =  unitigGraph.lengths[unitig] - 2.0*unitigGraph.overlapLength + readLength
                    
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
    rho = 1.0e7
    
    i = 0
    
    while i < 1000:
   
        setWeightsD(augmentedBiGraph, gaussianNLL_D, sedges, xVals, Lengths)
    
        path = nx.bellman_ford_path(augmentedBiGraph, 'source+', 'sink+', weight='dweight')
    
        weight = evalPathWeight(augmentedBiGraph, path, 'dweight')

        #rpath = nx.bellman_ford_path(augmentedBiGraph, 'source+', 'sink+', weight='rweight')
    
        #rweight = evalPathWeight(augmentedBiGraph, path, 'rweight')

        #if abs(weight) > abs(rweight):
            
        pflow = -weight/rho

        if pflow > 0.:
            addFlowPath(augmentedBiGraph, path, pflow)
            print("pflow: " +  str(pflow))
        
        #else:
            
         #   rflow = -rweight/rho
                
          #  if rflow < 0.:
           #     addFlowPath(augmentedBiGraph, path, rflow)
            #    print("rflow: " +  str(rflow))

        (dF, F) = evalDF(augmentedBiGraph, gaussianNLL_F, gaussianNLL_D, sedges, xVals, Lengths)

        print(str(i) + "," + str(dF) + "," + str(F))

        i+=1

    import ipdb; ipdb.set_trace()
    #nx.write_graphml(unitigGraph.augmentedUnitigBiGraphS,"test.graphml")

    #def dijkstra_path(G, source, target, weight='weight')

if __name__ == "__main__":
    main(sys.argv[1:])
