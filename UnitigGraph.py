import operator
import sys, getopt
import os
import pandas as p
import numpy as np
import scipy.stats as ss
import scipy as sp
import scipy.misc as spm
import scipy.special as sps
import math
import argparse

import networkx as nx

import collections
from collections import deque
from collections import defaultdict
from collections import Counter
from numpy.random import RandomState

from Bio import SeqIO
from Bio import pairwise2

MAX_EVALUE = 0.01

from Utils import reverseComplement
from Utils import convertNodeToName
from Utils import convertNameToNode2
from Utils import read_coverage_file

mapDirn = {'True' : "+", 'False' : "-"}

def dijkstra(graph, source, sink):

    try:
        (length,path) = nx.bidirectional_dijkstra(graph,source,sink)
        return (length,path) 
    except:
        return False


class UnitigGraph():
    """Creates unitig graph"""

    def __init__(self,kmerLength):
        self.overlapLength = kmerLength - 1
        self.sequences = {}
        self.lengths = {}
        self.overlaps = defaultdict(lambda: defaultdict(list))
        self.undirectedUnitigGraph = nx.Graph()
        self.unitigs = []
        self.N = 0
        self.forward = {}
        self.start = {}
        self.NC = 0
        self.covMap = {}
    
    @classmethod
    def loadGraph(cls,unitigFile, covFile, kmerLength):
    
        unitigGraph = cls(kmerLength)
    
        for record in SeqIO.parse(unitigFile, "fasta"):
            unitigGraph.sequences[record.id] = str(record.seq)
            unitigGraph.lengths[record.id] = len(record.seq)
            
            unitigGraph.undirectedUnitigGraph.add_node(record.id)
            
            desc_fields = record.description.split()
            #remove first four
            del desc_fields[:4]
            
            for link in desc_fields:
                link_toks = link.split(':')
                
                linkTo = link_toks[2]
                
                start = True
                if link_toks[1] == '-':
                    start = False
                
                end = True
                if link_toks[3] == '-':
                    end = False
                
                unitigGraph.overlaps[record.id][linkTo].append((start,end))
                
                unitigGraph.undirectedUnitigGraph.add_edge(record.id,linkTo)
        unitigGraph.unitigs = unitigGraph.undirectedUnitigGraph.nodes()
        unitigGraph.N = nx.number_of_nodes(unitigGraph.undirectedUnitigGraph)
        unitigGraph.NC = nx.number_connected_components(unitigGraph.undirectedUnitigGraph)
        unitigGraph.covMap = read_coverage_file(covFile)
        unitigGraph.createDirectedBiGraph()
        return unitigGraph
        
    def createUndirectedGraphSubset(self,unitigList):
        newGraph = UnitigGraph(self.overlapLength + 1)
    
        newGraph.overlapLength = self.overlapLength
        
        newGraph.undirectedUnitigGraph = self.undirectedUnitigGraph.subgraph(unitigList) 
        
        newGraph.sequences = {k: self.sequences[k] for k in unitigList}
        
        newGraph.lengths = {k: self.lengths[k] for k in unitigList}
        
        newGraph.overlaps = {k: self.overlaps[k] for k in unitigList}
        
        newGraph.covMap = {k: self.covMap[k] for k in unitigList}
        
        newGraph.N = nx.number_of_nodes(newGraph.undirectedUnitigGraph)
        newGraph.unitigs = newGraph.undirectedUnitigGraph.nodes()
        newGraph.forward = {}
        newGraph.start = {}
        newGraph.NC = nx.number_connected_components(newGraph.undirectedUnitigGraph)
        newGraph.createDirectedBiGraph()
        return newGraph
    
    def setDirectionOrder(self, unitig_order):
        
        self.end = {}
        
        for unitig, order in unitig_order.items():
            (hitStart,hitEnd,dirn) = order
            
            if hitStart is not None:
                self.start[unitig] = hitStart
            
            if hitEnd is not None:
                self.end[unitig] = hitEnd
            
            if dirn is not None:
                self.forward[unitig] = dirn

    def isSourceSink(self, unitig):
      
        allLinks = []
        for outNode,links in self.overlaps[unitig].items():
            
            for link in links:
                allLinks.append(link[0])
        if len(allLinks) > 0:    
            if all(x == allLinks[0] for x in allLinks):
                return allLinks[0]
            else:
                return None
        else:
            return None
            
    def getSourceSinks(self):
    
        sources = []
        sinks = []
        
        for unitig in self.undirectedUnitigGraph.nodes():
            sourceSink = self.isSourceSink(unitig)
            if sourceSink is not None and unitig in self.forward:
                nodeName = convertNodeToName((unitig,self.forward[unitig]))
                if sourceSink == self.forward[unitig]:
                    sources.append(nodeName)
                else:
                    sinks.append(nodeName)
                    
        return (sources,sinks)
    
    def getUnreachableBiGraph(self,source_names,sink_names):
    
        unreachablePlusMinus = []
        nodeReachable = {}
        if len(source_names) > 0 and len(sink_names) > 0: 
            for unitig in self.unitigs:
        
                nodePlus  = (unitig, True)
                nodeMinus = (unitig, False)
            
                nodePlusName = convertNodeToName(nodePlus)
                nodeMinusName = convertNodeToName(nodeMinus)
            
                reachablePlus = False
                if any(dijkstra(self.directedUnitigBiGraph,source,nodePlusName) for source in source_names):
                    if any(dijkstra(self.directedUnitigBiGraph,nodePlusName,sink) for sink in sink_names):
                        reachablePlus = True
                    
                reachableMinus = False
                if any(dijkstra(self.directedUnitigBiGraph,source,nodeMinusName) for source in source_names):
                    if any(dijkstra(self.directedUnitigBiGraph,nodeMinusName,sink) for sink in sink_names):
                        reachableMinus = True
            
                if reachablePlus == False:
                    unreachablePlusMinus.append(nodePlusName)
                    
                if reachableMinus == False:
                    unreachablePlusMinus.append(nodeMinusName)
                
                if reachableMinus or reachablePlus:
                    nodeReachable[unitig] = True
                else:
                    nodeReachable[unitig] = False
                    
        else:
            for unitig in unitigSubGraph.unitigs:
                nodeReachable[unitig] = True
                
        return (nodeReachable, unreachablePlusMinus)
        
    def selectSourceSinks(self, sub_unitig_order, dFrac):
        
        self.setDirectionOrder(sub_unitig_order)
    
        (sources, sinks) = self.getSourceSinks()
        
        sourceSinkLength = defaultdict(dict)
        maxSourceSink = 0.
        maxSource = None
        maxSink = None
        for source in sources:
            pathLengths = nx.single_source_dijkstra_path_length(self.directedUnitigBiGraph,source)
            
            for sink in sinks:
                if sink in pathLengths:
                    sourceSinkLength[source][sink] = pathLengths[sink]
                    if pathLengths[sink] > maxSourceSink:
                        maxSourceSink = pathLengths[sink]
                        maxSource = source
                        maxSink = sink
        if maxSource is not None:
            source_select = {maxSource}
        else:
            source_select = {}
        
        if maxSink is not None:
            sink_select = {maxSink}
        else:
            sink_select = {}
        
        for source in sources:
            for sink in sinks:
                if sink in sourceSinkLength[source] and sourceSinkLength[source][sink] > dFrac*maxSourceSink:
                    source_select.add(source)
                    sink_select.add(sink)
        
        if len(source_select) > 0:
            source_list = list(map(convertNameToNode2, source_select))
        else:
            source_list = []
        if len(sink_select) > 0:
            sink_list = list(map(convertNameToNode2, sink_select))
        else:
            sink_list = []
        return (source_list,sink_list)
        
    def createDirectedBiGraph(self):
        self.directedUnitigBiGraph = nx.DiGraph()
        
        for node in self.undirectedUnitigGraph.nodes():
            nodePlus  = (node, True)
            nodeMinus = (node, False)
            
            nodePlusName = convertNodeToName(nodePlus)
            nodeMinusName = convertNodeToName(nodeMinus)
        
            if nodePlusName not in self.directedUnitigBiGraph:
                self.directedUnitigBiGraph.add_node(nodePlusName)
            
            if nodeMinusName not in self.directedUnitigBiGraph:
                self.directedUnitigBiGraph.add_node(nodeMinusName)
        
            #get all links outgoing given +ve direction on node
            for outnode, dirns in self.overlaps[node].items():

                for dirn in dirns:
                    (start,end) = dirn
            
                    #add outgoing positive edge
                
                    if start:
                        addEdge = (nodePlus, (outnode, end)) 
    
                    else:
                        #reverse as incoming edge
                        addEdge = ((outnode,not end), (node, True))
                
                    edgeName = convertNodeToName(addEdge)
                    nodePlusOutName = convertNodeToName((outnode,end))
                    lengthPlus = self.lengths[node] - self.overlapLength
                    nodeMinusOutName = convertNodeToName((outnode,not end))
                    lengthMinus = self.lengths[outnode] - self.overlapLength
                    
                    if start:
                        self.directedUnitigBiGraph.add_edge(nodePlusName, nodePlusOutName, weight=lengthPlus)
                    else:
                        self.directedUnitigBiGraph.add_edge(nodeMinusOutName, nodePlusName, weight=lengthMinus)        
            
                    #add negative edges
                    if start:
                        #reverse as incoming edge
                        addEdge = ((outnode, not end),(nodeMinus))
                    else:    
                        addEdge = (nodeMinus, (outnode, end)) 
                
                    edgeName = convertNodeToName(addEdge)
                
                    if start:
                        self.directedUnitigBiGraph.add_edge(nodeMinusOutName, nodeMinusName, weight=lengthMinus)
                    else:
                        self.directedUnitigBiGraph.add_edge(nodeMinusName, nodePlusOutName, weight=lengthPlus)
    
    def writeFlipFlopFiles(self, newsource, newsink, assGraph):
    
        biGraph = self.directedUnitigBiGraph.copy()
        biGraphNodes = nx.topological_sort(biGraph)
         
        source_names = [convertNodeToName(source) for source in newsource] 
        sink_names = [convertNodeToName(sink) for sink in newsink]
        (nodeReachable, unreachablePlusMinus) = self.getUnreachableBiGraph(source_names,sink_names)
  
        for unreachable in unreachablePlusMinus:
            biGraph.remove_node(unreachable)
        
        #remove unreachable nodes
        nx.write_graphml(biGraph,"biGraph.graphml")
  
        nodeMap = {}
        with open("CovF.csv", 'w') as covf_file:
        
            n = 1
            for node in biGraph:
                nodeMap[node] = n
                noded = node[:-1]
                covList = assGraph.covMap[noded].tolist()
                outString = ",".join([str(x) for x in covList])
                covf_file.write(str(n) + "," + node + "," + str(assGraph.adjLengths[noded]) + "," + outString + "\n")
                
                n = n + 1
        edges = []
        nNodes=len(biGraph.nodes())
        for edge in biGraph.edges_iter():
            edges.append(edge)
        
        edges = sorted(edges)
        with open("diGraph.csv", 'w') as graph_file:
            graph_file.write(str(nNodes)+"\n")
            
            for edge in edges:
                graph_file.write(str(nodeMap[edge[0]]) +"," +str(nodeMap[edge[1]]) + "\n")
         
    def getUnitigWalk(self, walk):
    
        newUnitig = ""
        bFirst = True
        for unitigd in walk:
            unitig = unitigd[:-1]
            dirn = unitigd[-1:] 
            
            if dirn == '+':
                direction = True
            else:
                direction = False

            seq = self.sequences[unitig]
            
            if direction == False:
                seq = reverseComplement(seq)
            
            if bFirst == False:
                seq = seq[self.overlapLength:]
            
            newUnitig = newUnitig + seq
    
            bFirst = False
    
        return newUnitig

    def addSourceSink(self, factorGraph, sources, sinks):
    
        #add sink factor and dummy flow out
        self.sinkNode = 'sink+'
        if self.sinkNode not in self.factorGraph:
            self.factorGraph.add_node(self.sinkNode, factor=True, code=('sink',True))
        
        node_code = (('sink',True),('infty',True))
        self.sinkEdge = self.convertNodeToName(node_code)
        self.factorGraph.add_node(self.sinkEdge, factor=False, code=node_code)
        self.factorGraph.add_edge(self.sinkNode, self.sinkEdge)
        
        #add source factor and dummy flow out
        self.sourceNode = 'source+'
        if self.sourceNode not in self.factorGraph:
            self.factorGraph.add_node(self.sourceNode, factor=True, code=('source',True))
        
        node_code = (('zero',True),('source',True))
        self.sourceEdge = self.convertNodeToName(node_code)
        self.factorGraph.add_node(self.sourceEdge, factor=False, code=node_code)
        self.factorGraph.add_edge(self.sourceEdge, self.sourceNode)
        
        for (sinkid,dirn) in sinks:
            
            node_code = ((sinkid,dirn),('sink',True))
            
            edgeName = self.convertNodeToName(node_code)
            
            sinkName = self.convertNodeToName((sinkid,dirn))
            
            self.factorGraph.add_node(edgeName, factor=False, code=node_code)
            
            self.factorGraph.add_edge(sinkName,edgeName)
            
            self.factorGraph.add_edge(edgeName,self.sinkNode)
        
        for (sourceid,dirn) in sources:
            
            node_code = (('source',True),(sourceid,dirn))
            
            edgeName = self.convertNodeToName(node_code)
            
            sourceName = self.convertNodeToName((sourceid,dirn))
            
            self.factorGraph.add_node(edgeName, factor=False, code=node_code)
            
            self.factorGraph.add_edge(edgeName,sourceName)
            
            self.factorGraph.add_edge(self.sourceNode,edgeName)

