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
import gfapy
import networkx as nx

import collections
from collections import deque
from collections import defaultdict
from collections import Counter
from numpy.random import RandomState

from Bio import SeqIO
from Bio import pairwise2

MAX_EVALUE = 0.01

from Utils.UtilsFunctions import reverseComplement
from Utils.UtilsFunctions import convertNodeToName
from Utils.UtilsFunctions import convertNameToNode2
from Utils.UtilsFunctions import read_coverage_file

mapDirn = {'True' : "+", 'False' : "-"}

def dijkstra(graph, source, sink):
    """Wrapper for networkx shortest path algorithm"""
    try:
        (length,path) = nx.bidirectional_dijkstra(graph,source,sink)
        return (length,path) 
    except:
        return False

def reverseDirn(node):
    """Reverses node direction"""
    if node[-1] == '+':
        node[-1] == '-'
    elif node[-1] == '+':
        node[-1] == '+'

class UnitigGraph():
    """Creates unitig graph"""

    def __init__(self,kmerLength,overlapLength):
        """Empty UnitigGraph object"""
        self.overlapLength = overlapLength # overlap length for unitig assume fixed
        self.kmerLength = kmerLength # de Bruijn graph kmer length
        self.sequences = {}
        self.lengths = {}
        self.overlaps = defaultdict(lambda: defaultdict(list)) # unitig overlaps as 2d dict of lists
        self.undirectedUnitigGraph = nx.Graph() # undirected graph representation
        self.unitigs = [] #list of untig names
        self.N = 0 # number of unitigs
        self.NC = 0 #number of components
        self.covMap = {} # coverages across samples may be null
        self.KC = {} # kmer support
        self.directedUnitigBiGraphS =  None
        self.directedUnitigBiGraph =  None
         
    @classmethod
    def loadGraph(cls,unitigFile, kmerLength, overlapLength = None, covFile = None):
        """Creates graphs from a Bcalm output file"""
        if overlapLength == None:
            overlapLength = kmerLength - 1 # if overlapLength not provided assume kmerLength - 1
        unitigGraph = cls(kmerLength, overlapLength)
    
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
        unitigGraph.unitigs = list(unitigGraph.undirectedUnitigGraph.nodes())
        unitigGraph.N = nx.number_of_nodes(unitigGraph.undirectedUnitigGraph)
        unitigGraph.NC = nx.number_connected_components(unitigGraph.undirectedUnitigGraph)
        if covFile is not None:
            unitigGraph.covMap = read_coverage_file(covFile)
        unitigGraph.createDirectedBiGraph()
        return unitigGraph
    
    @classmethod
    def loadGraphFromGfaFile(cls,gfaFile, kmerLength = None, covFile = None, tsvFile = False, bRemoveSelfLinks = False):
        """Creates graphs from a GFA file and an optional coverage file as csv without header"""
        gfa = gfapy.Gfa.from_file(gfaFile)
    
        if kmerLength is None:
            if hasattr(gfa.header, 'kk') and gfa.header.kk is not None:
                kmerLength = int(gfa.header.kk)
            else:
                raise ValueError("Problem setting kmerLength from gfa")
                
        unitigGraph = cls(kmerLength, kmerLength)
        
        unitigGraph.N=len(gfa.segments)
        
        # unitig lengths

        for seg in gfa.segments:
            id = seg.name
            unitigGraph.lengths[id] = len(seg.sequence)
            unitigGraph.sequences[id] = str(seg.sequence)
            unitigGraph.undirectedUnitigGraph.add_node(id)
            unitigGraph.KC[id] = int(seg.KC)
            
        for edge in gfa.edges:
        
            start = True
            end = True
            if edge.from_orient == "-":
                start = False
            
            if edge.to_orient == "-":
                end = False
            
            if not bRemoveSelfLinks or edge.to_segment.name != edge.from_segment.name:
            
                unitigGraph.overlaps[edge.from_segment.name][edge.to_segment.name].append((start,end))

                unitigGraph.overlaps[edge.to_segment.name][edge.from_segment.name].append((not end,not start))
                
                unitigGraph.undirectedUnitigGraph.add_edge(edge.from_segment.name,edge.to_segment.name)
        
        unitigGraph.unitigs = list(unitigGraph.undirectedUnitigGraph.nodes())
        unitigGraph.N = nx.number_of_nodes(unitigGraph.undirectedUnitigGraph)
        unitigGraph.NC = nx.number_connected_components(unitigGraph.undirectedUnitigGraph)
        if covFile is not None:
            unitigGraph.covMap = read_coverage_file(covFile,tsvFile)
        else:
            unitigGraph.covMap = None
        unitigGraph.createDirectedBiGraph()
        return unitigGraph
    
    @classmethod
    def combineGraphs(cls,geneOrder, unitigGraphList, sourcesLists, sinksLists):
    
        assert(len(unitigGraphList) > 0)


        newKmerLength = next(iter(unitigGraphList.values())).kmerLength
        newOverLength = next(iter(unitigGraphList.values())).overlapLength
        
        combinedGraph = cls(newKmerLength, newOverLength)

          
        for gene in geneOrder:
            
            unitigGraph = unitigGraphList[gene]
            assert (unitigGraph.kmerLength == newKmerLength)
            combinedGraph.sequences.update(unitigGraph.sequences)
            combinedGraph.lengths.update(unitigGraph.lengths)
            combinedGraph.overlaps.update(unitigGraph.overlaps)
            
            combinedGraph.undirectedUnitigGraph = nx.compose(combinedGraph.undirectedUnitigGraph,unitigGraph.undirectedUnitigGraph) 
            combinedGraph.unitigs = combinedGraph.unitigs + unitigGraph.unitigs
            combinedGraph.N += unitigGraph.N
            combinedGraph.NC += unitigGraph.NC
            combinedGraph.covMap.update(unitigGraph.covMap)
            combinedGraph.KC.update(unitigGraph.KC)
        
        if combinedGraph.covMap is not None:
            S = combinedGraph.covMap[combinedGraph.unitigs[0]].shape[0]
        
        #Now add connections
        
        c = 0
        for gene, nextGene in zip(geneOrder,geneOrder[1:]): 
            sinkList = sinksLists[gene]
            nextSourceList = sourcesLists[nextGene]
            
            connect = 'connect_0' + str(c)
            
            combinedGraph.unitigs.append(connect)
            combinedGraph.sequences[connect] = 'N'*combinedGraph.overlapLength
        
            combinedGraph.lengths[connect] = combinedGraph.overlapLength
            combinedGraph.N += 1
            
            if combinedGraph.KC is not None:
                combinedGraph.KC[connect] = 0
            if combinedGraph.covMap is not None:
                combinedGraph.covMap[connect] = np.zeros(S)
            
            for (sinkUnitig, sinkDirn) in sinkList:
                combinedGraph.undirectedUnitigGraph.add_edge(sinkUnitig,connect)
                
                combinedGraph.overlaps[sinkUnitig][connect].append((sinkDirn,True))
                combinedGraph.overlaps[connect][sinkUnitig].append((False, not sinkDirn))
                
            for (sourceUnitig, sourceDirn) in nextSourceList:
                combinedGraph.undirectedUnitigGraph.add_edge(connect, sourceUnitig)
            
                combinedGraph.overlaps[connect][sourceUnitig].append((True, sourceDirn))
                combinedGraph.overlaps[sourceUnitig][connect].append((not sourceDirn, False))
            
            c = c + 1
            
        return combinedGraph
    

    
    @classmethod
    def getReachableSubset(cls,forwardGraph,coreNodes):
        reverseGraph = forwardGraph.reverse()
        reachable = []
        setCoreNodes = set(coreNodes)
        for node in forwardGraph.nodes():
            if node in setCoreNodes:
                reachable.append(node)
            else:
                nDes = nx.descendants(forwardGraph,node)
                nAnc = nx.descendants(reverseGraph,node)
    
                if len(nDes & setCoreNodes) > 0 and len(nAnc & setCoreNodes) > 0:
                    reachable.append(node)
        
        return reachable 
    
    @classmethod
    def getLowestCommonDescendant(cls,forwardGraph,coreNodes):
    
        assert nx.is_directed_acyclic_graph(forwardGraph), "Graph has to be acyclic and directed."

        # get descendentgs of all

        core_descendents = []
        for coreNode in coreNodes:
            tDec = nx.descendants(forwardGraph,coreNode)
            tDec.add(coreNode)
            core_descendents.append(tDec)
            
        common_descendants = set.intersection(*core_descendents)
    
        allPaths = nx.all_pairs_dijkstra_path_length(forwardGraph)

        allSSDict = defaultdict(dict)       
  
        for sourcePaths in allPaths:
            source = sourcePaths[0]
    
            if source in coreNodes:

                for sink, length in sourcePaths[1].items(): 
                    
                    if sink in common_descendants:
                        allSSDict[source][sink] = length
                        
        # get sum of path lengths
        if len(common_descendants) > 0:
            sum_of_path_lengths = np.zeros((len(common_descendants)))
            common_descendants_list = list(common_descendants)
            for ii, c in enumerate(common_descendants_list):
            
                for coreNode in coreNodes:
                    sum_of_path_lengths[ii] += allSSDict[coreNode][c]
        
            minima = np.where(sum_of_path_lengths == np.min(sum_of_path_lengths))
    
            lca = [common_descendants_list[ii] for ii in minima[0]]
        else:
            lca = None
             
        return lca

    @classmethod
    def getLowestCommonDescendantG(cls,forwardGraph,coreNodes):
    
        allPaths = nx.all_pairs_dijkstra_path_length(forwardGraph)
        
        decCore = defaultdict(set)
        allSSDict = defaultdict(dict)  
               
        for sourcePaths in allPaths:
            source = sourcePaths[0]
    
            if source in coreNodes:
                for sink, length in sourcePaths[1].items(): 
                    decCore[source].add(sink)
                    allSSDict[source][sink] = length
                    
        # get descendentgs of all

        core_descendents = []
        for coreNode in coreNodes:
            tDec = decCore[coreNode]
            tDec.add(coreNode)
            core_descendents.append(tDec)
            
        common_descendants = set.intersection(*core_descendents)     

                        
        # get sum of path lengths
        if len(common_descendants) > 0:
            sum_of_path_lengths = np.zeros((len(common_descendants)))
            common_descendants_list = list(common_descendants)
            for ii, c in enumerate(common_descendants_list):
            
                for coreNode in coreNodes:
                    sum_of_path_lengths[ii] += allSSDict[coreNode][c]
        
            minima = np.where(sum_of_path_lengths == np.min(sum_of_path_lengths))
    
            lca = [common_descendants_list[ii] for ii in minima[0]]
        else:
            lca = None
             
        return lca

    
    def writeCovToCSV(self,fileName):
        if self.covMap is None:
            raise TypeError
    
        with open(fileName, 'w') as covFile:
    
            for unitig in self.unitigs:
                cString = ",".join([str(x) for x in self.covMap[unitig].tolist()])
                
                covFile.write(unitig + "," + cString + "\n")
    
    
    def writeToGFA(self,fileName):
        """Writes UnitigGraph to GFA file"""
        with open(fileName, 'w') as gfaFile:
    
            for unitig in self.unitigs:
                sVals = ['S',str(unitig),self.sequences[unitig],'LN:i:'+str(self.lengths[unitig])]
                if self.KC[unitig] is not None:
                    sVals.append('KC:i:' + str(self.KC[unitig]))
                sString = '\t'.join(sVals)
                gfaFile.write(sString + '\n')
        
            #just use this to only output list once
            tempHash = {}
            for unitig in self.unitigs:
                for outUnitig,linkList in self.overlaps[unitig].items():
                    hashTag = unitig + "_" + outUnitig
                    rTag = outUnitig + "_" + unitig
                    if hashTag not in tempHash and rTag not in tempHash:
                   
                        for link in linkList:
                            if link[0]:
                                link1 = '+'
                            else:
                                link1 = '-'
                                
                            if link[1]:
                                link2 = '+'
                            else:
                                link2 = '-'
                            
                            lVals = ['L',str(unitig),link1,outUnitig,link2,str(self.overlapLength)+'M']
                            
                            
                            lString = '\t'.join(lVals)
                            
                            gfaFile.write(lString + '\n')
                        
                        tempHash[hashTag] = 1
                        tempHash[rTag]    = 1
    
    def createUndirectedGraphSubset(self,unitigList):
        newGraph = UnitigGraph(self.kmerLength,self.overlapLength)
        
        newGraph.undirectedUnitigGraph = self.undirectedUnitigGraph.subgraph(unitigList) 
        
        newGraph.sequences = {k: self.sequences[k] for k in unitigList}
        
        newGraph.lengths = {k: self.lengths[k] for k in unitigList}
       
        newGraph.overlaps = defaultdict(dict)
        for k in unitigList:
            for l, links in self.overlaps[k].items():
                if l in unitigList:
                    newGraph.overlaps[k][l] = links
        
        if self.covMap is not None:
            newGraph.covMap = {k: self.covMap[k] for k in unitigList}
        else:
            newGraph.covMap = None
        
        if self.KC: 
            newGraph.KC = {k: self.KC[k] for k in unitigList}
        
        newGraph.N = nx.number_of_nodes(newGraph.undirectedUnitigGraph)
        newGraph.unitigs = list(newGraph.undirectedUnitigGraph.nodes())
        newGraph.NC = nx.number_connected_components(newGraph.undirectedUnitigGraph)
        newGraph.createDirectedBiGraph()
        return newGraph
    

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
            
    def isSourceSinkIsolated(self, unitig):
      
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
            return True

            
    def isIsolated(self, unitig):
      
        allLinks = []
        for outNode,links in self.overlaps[unitig].items():
            
            for link in links:
                allLinks.append(link[0])
        
        if len(allLinks) > 0:    
            return False
        else:
            return True
            
            
    
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
    
    def selectAllSourceSinks(self):
    
        sources = []
        sinks = []
        
        sourceSinks = []
        for unitig in self.undirectedUnitigGraph.nodes():
            sourceSink = self.isSourceSink(unitig)
            
            if sourceSink is not None:
                sourceSinks.append(unitig)
            
        #get all path lengths
        self.createDirectedBiGraph()        

        allPaths = nx.all_pairs_dijkstra_path_length(self.directedUnitigBiGraph)
        allSSDict = defaultdict(dict)       
        maxSourceSink = 0
        maxSource = None
        maxSink = None
        for sourcePaths in allPaths:
            source = sourcePaths[0]
            unitig = source[:-1]
            if unitig in sourceSinks:

                for sink, length in sourcePaths[1].items(): 
                    sinkUnitig = sink[:-1]
                    if sinkUnitig in sourceSinks:
                        allSSDict[source][sink] = length
                        if length > maxSourceSink:
                            maxSourceSink = length
                            maxSource = source
                            maxSink = sink

        #set first node as source
        sUnitig = sourceSinks.pop(0)
        bForward = True
        
        if len(list(self.directedUnitigBiGraph.neighbors(sUnitig + "+"))) > 0:  
            sources.append(sUnitig + "+")
        else:
            sources.append(sUnitig + "-")
       
        #add longest alternative as sink
        newSink = None
        maxSink = sys.float_info.max
        for sourceSink in sourceSinks:
            ssPlus  = sourceSink + "+"
            ssMinus = sourceSink + "-"

            if ssPlus in allSSDict[sources[0]]:
                if allSSDict[sources[0]][ssPlus] < maxSink:
                    maxSink = allSSDict[sources[0]][ssPlus]
                    newSink = ssPlus
            
            if ssMinus in allSSDict[sources[0]]:
                if allSSDict[sources[0]][ssMinus] < maxSink:
                    maxSink = allSSDict[sources[0]][ssMinus]
                    newSink = ssMinus
        
        sinks.append(newSink)
        sourceSinks.remove(newSink[:-1])    
 
        for sourceSink in sourceSinks:
            ssPlus  = sourceSink + "+"
            ssMinus = sourceSink + "-"            
            
            maxSinkPlus = 0.
            for source in sources:
                if ssPlus in allSSDict[source]:
                    if allSSDict[source][ssPlus] > maxSinkPlus:
                        maxSinkPlus = allSSDict[source][ssPlus]
            
            maxSinkMinus = 0.0
            for source in sources:
                if ssMinus in allSSDict[source]:
                    if allSSDict[source][ssMinus] > maxSinkMinus:
                        maxSinkMinus = allSSDict[source][ssMinus]
            
            maxSourcePlus = 0.0
            for sink in sinks:
                if sink in allSSDict[ssPlus]:
                    if allSSDict[ssPlus][sink] > maxSourcePlus:
                        maxSourcePlus = allSSDict[ssPlus][sink]
            
            maxSourceMinus = 0.0
            for sink in sinks:
                if sink in allSSDict[ssMinus]:
                    if allSSDict[ssMinus][sink] > maxSourceMinus:
                        maxSourceMinus = allSSDict[ssMinus][sink]
            
            maxSink   = 0.
            maxSource = 0.
            bSinkPlus = True
            if maxSinkPlus > maxSinkMinus:
                maxSink = maxSinkPlus
            else:
                maxSink = maxSinkMinus
                bSinkPlus = False
            
            bSourcePlus = True
            if maxSourcePlus > maxSourceMinus:
                maxSource = maxSourcePlus
            else:
                maxSource = maxSourceMinus
                bSourcePlus = False
            
            if maxSink > maxSource:
                if bSinkPlus:
                    sinks.append(ssPlus)
                else:
                    sinks.append(ssMinus)
            else:
                if bSourcePlus:
                    sources.append(ssPlus)
                else:
                    sources.append(ssMinus)    
        
        if len(sources) > 0:
            source_list = list(map(convertNameToNode2, sources))
        else:
            source_list = []
        if len(sinks) > 0:
            sink_list = list(map(convertNameToNode2, sinks))
        else:
            sink_list = []
        
        return (source_list,sink_list)

    def isReachable(self, target, sources, pathDict):
        hit = None
        for source in sources:
            targetPlus = target + "+"
            
            if targetPlus in pathDict[source]:
                hit = targetPlus
                break
        
            targetMinus = target + "-"
            
            if targetMinus in pathDict[source]:
                hit = targetMinus
                break
        
        return hit
    
    def isReachableReverse(self, target,sinks, pathDict):
        
        hit = None
        for sink in sinks:
            targetPlus = target + "+"
            
            if sink in pathDict[targetPlus]:
                hit = targetPlus
                break
        
            targetMinus = target + "-"
            
            if sink in pathDict[targetMinus]:
                hit = targetMinus
                break
        
        return hit
        
            
    def selectSourceSinksStops(self, stops, deadends, minLength=0.):
    
        if self.N == len(stops):
            return (stops,stops)

        sources = []
        sinks = []

        deadEndSet = set(deadends)        
        sourceSinks = []
        for unitig in self.undirectedUnitigGraph.nodes():
            sourceSink = self.isSourceSinkIsolated(unitig)
            
            if sourceSink is not None and sourceSink not in deadEndSet:
                sourceSinks.append(unitig)
            
        #get all path lengths
        self.createDirectedBiGraph()
        #define sinks as stop tips and tips reachable from from stops on forward graph
        
        stopMap = {}
        stopMapUnitig = {}
        for stop in stops:
            stopMap[stop[0]] = stop
            stopMapUnitig[stop[0][:-1]] = stop[0]
        allPaths = nx.all_pairs_dijkstra_path_length(self.directedUnitigBiGraph)
        allSSDict = defaultdict(dict)       

        stopsSet = set(stopMap.keys())
        sourceSinksStops = list(set(sourceSinks).union(stopsSet))

        for sourcePaths in allPaths:
            source = sourcePaths[0]
            unitig = source[:-1]
            if unitig in sourceSinksStops:

                for sink, length in sourcePaths[1].items(): 
                    sinkUnitig = sink[:-1]
                    if sinkUnitig in sourceSinksStops:
                        allSSDict[source][sink] = length

        sinks = []
        sinkUnitigs = set()
        sourceUnitigs = set()
        stopNames = [convertNodeToName(x) for x in stops] 
        
        for sourceSink in sourceSinks:
            if sourceSink in stopMap:
                if self.isIsolated(sourceSink) == False:
                    sinks.append(convertNodeToName(stopMap[sourceSink]))
                    sinkUnitigs.add(sourceSink)
                else:
                    if self.lengths[sourceSink] > minLength:
                        sinks.append(convertNodeToName(stopMap[sourceSink]))
                        sinkUnitigs.add(sourceSink)
            else:
                hit = self.isReachable(sourceSink,stopNames,allSSDict)
                if hit is not None:
                    sinks.append(hit)
                    sinkUnitigs.add(sourceSink)
        
        sources = []
        for sourceSink in sourceSinks:
            if sourceSink not in sinkUnitigs:
                hit = self.isReachableReverse(sourceSink,sinks,allSSDict)
     
                if hit is not None:
                    sources.append(hit)
                    sourceUnitigs.add(sourceSink)
                
        #now add remainder if reachable in reverse
        for sourceSink in sourceSinks:
            if sourceSink not in sinkUnitigs and sourceSink not in sourceUnitigs:
                hit = self.isReachable(sourceSink,sources,allSSDict)

                if hit is not None:
                    sinks.append(hit)
                    sinkUnitigs.add(sourceSink)

        #add any isolated singletons here
        sinkMapUnitigs = {x[:-1]:x for x in sinks}
        sourceMapUnitigs = {x[:-1]:x for x in sources}
        
        for sourceSink in sourceSinks:
            if self.isIsolated(sourceSink) and self.lengths[sourceSink] > minLength:
            
                if sourceSink not in sinkUnitigs and sourceSink not in sourceUnitigs:
                    sinks.append(sourceSink + "+")
                    sinkUnitigs.add(sourceSink)
                    sources.append(sourceSink + "+")
                    sourceUnitigs.add(sourceSink)
                elif sourceSink not in sourceUnitigs:
                    
                    sink = sinkMapUnitigs[sourceSink]
                    dirn = sink[-1]
                    if dirn == '+':
                        sources.append(sourceSink + "+")
                    else:
                        sources.append(sourceSink + "-")
                elif sourceSink not in sinkUnitigs:
                
                    source = sourceMapUnitigs[sourceSink]
                    dirn = source[-1]
                    if dirn == '+':
                        sinks.append(sourceSink + "+")
                    else:
                        sinks.append(sourceSink + "-")
                        
                    
        if len(sources) > 0:
            source_list = list(map(convertNameToNode2, sources))
        else:
            source_list = []
        if len(sinks) > 0:
            sink_list = list(map(convertNameToNode2, sinks))
        else:
            sink_list = []
        
        return (source_list,sink_list)
        
    
    def selectSourceSinks(self, dFrac):
        
        if self.N == 1:
            return ([(self.unitigs[0],True)],[(self.unitigs[0],True)])
        
        
        sources = []
        sinks = []
        
        isolated = []
        
        for unitig in self.undirectedUnitigGraph.nodes():
            if self.isIsolated(unitig) and self.lengths[unitig] > 200:
                isolated.append(unitig)
        
        sourceSinks = []
        for unitig in self.undirectedUnitigGraph.nodes():
            sourceSink = self.isSourceSink(unitig)
            
            if sourceSink is not None:
                sourceSinks.append(unitig)
            
        #get all path lengths
        if len(sourceSinks) > 0:
            self.createDirectedBiGraph()        

            allPaths = nx.all_pairs_dijkstra_path_length(self.directedUnitigBiGraph)
            allSSDict = defaultdict(dict)       
            maxSourceSink = 0
            maxSource = None
            maxSink = None
            for sourcePaths in allPaths:
                source = sourcePaths[0]
                unitig = source[:-1]
                if unitig in sourceSinks:

                    for sink, length in sourcePaths[1].items(): 
                        sinkUnitig = sink[:-1]
                        if sinkUnitig in sourceSinks:
                            allSSDict[source][sink] = length
                            if length > maxSourceSink:
                                maxSourceSink = length
                                maxSource = source
                                maxSink = sink
            
            sources = [maxSource]
            sourceTigs = [maxSource[:-1]]
            sinks = [maxSink]
            sinkTigs = [maxSink[:-1]]

            for sourceSink in sourceSinks:
                if sourceSink not in sourceTigs and sourceSink not in sinkTigs:
                    ssPlus = sourceSink + "+"
                    ssMinus = sourceSink + "-"
                
                    maxSourcePlus = 0.
                    maxSourceMinus = 0.

                    if ssPlus in allSSDict: 
                        for sink in sinks:
                            if sink in allSSDict[ssPlus]:
                                if allSSDict[ssPlus][sink] > maxSourcePlus:
                                    maxSourcePlus = allSSDict[ssPlus][sink]
                 
                    if ssMinus in allSSDict:
                        for sink in sinks:
                            if sink in allSSDict[ssMinus]:
                                if allSSDict[ssMinus][sink] > maxSourceMinus:
                                    maxSourceMinus = allSSDict[ssMinus][sink]
                
                    bSourcePlus = True
                    sourceMax = maxSourcePlus
                    if maxSourcePlus < maxSourceMinus:
                        bSourcePlus = False
                        sourceMax = maxSourceMinus
                
                    maxSinkPlus = 0.0
                    maxSinkMinus = 0.0

                    for source in sources:
                        if ssPlus in allSSDict[source]:
                            if allSSDict[source][ssPlus] > maxSinkPlus:
                                maxSinkPlus = allSSDict[source][ssPlus]

                    for source in sources:
                        if ssMinus in allSSDict[source]:
                            if allSSDict[source][ssMinus] > maxSinkMinus:
                                maxSinkMinus = allSSDict[source][ssMinus]
                
                    sinkMax = maxSinkPlus
                    bSinkPlus = True
                    if maxSinkPlus < maxSinkMinus:
                        bSinkPlus = False
                        sinkMax = maxSinkMinus
                
                    if sourceMax > sinkMax:
                        if sourceMax > dFrac*maxSourceSink:
                            if bSourcePlus:
                                sources.append(ssPlus)
                            else:
                                sources.append(ssMinus)
                            sourceTigs.append(sourceSink)
                    else:
                        if sinkMax > dFrac*maxSourceSink:
                            if bSinkPlus:
                                sinks.append(ssPlus)
                            else: 
                                sinks.append(ssMinus)

                            sinkTigs.append(sourceSink)
        for isolate in isolated:
            sources.append(isolate + "+")
            sinks.append(isolate + "+")

        if len(sources) > 0:
            source_list = list(map(convertNameToNode2, sources))
        else:
            source_list = []
        if len(sinks) > 0:
            sink_list = list(map(convertNameToNode2, sinks))
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
                    if self.covMap is not None:
                        nodePlusSumCov = np.sum(self.covMap[node])*lengthPlus
                    else:
                        nodePlusSumCov = self.KC[node]
                        
                    nodeMinusOutName = convertNodeToName((outnode,not end))
                    lengthMinus = self.lengths[outnode] - self.overlapLength
                    
                    if self.covMap is not None:
                        nodeMinusSumCov = np.sum(self.covMap[outnode])*lengthMinus
                    else:
                        nodeMinusSumCov = self.KC[outnode]
                        
                    if start:
                        self.directedUnitigBiGraph.add_edge(nodePlusName, nodePlusOutName, weight=lengthPlus,covweight=nodePlusSumCov)
                    else:
                        self.directedUnitigBiGraph.add_edge(nodeMinusOutName, nodePlusName, weight=lengthMinus,covweight=nodeMinusSumCov)        
            
                    #add negative edges
                    if start:
                        #reverse as incoming edge
                        addEdge = ((outnode, not end),(nodeMinus))
                    else:    
                        addEdge = (nodeMinus, (outnode, end)) 
                
                    edgeName = convertNodeToName(addEdge)
                
                    if start:
                        self.directedUnitigBiGraph.add_edge(nodeMinusOutName, nodeMinusName, weight=lengthMinus,covweight=nodeMinusSumCov)
                    else:
                        self.directedUnitigBiGraph.add_edge(nodeMinusName, nodePlusOutName, weight=lengthPlus,covweight=nodePlusSumCov)
        #if node is sink need to add extra length
        for node in self.directedUnitigBiGraph.nodes():   
            if self.directedUnitigBiGraph.out_degree(node) == 0:
                for innode in self.directedUnitigBiGraph.predecessors(node):

                    newWeight = self.directedUnitigBiGraph[innode][node]['weight'] + self.lengths[node[:-1]]  
                    if self.covMap is not None:
                        newSum = self.directedUnitigBiGraph[innode][node]['covweight'] + self.lengths[node[:-1]]*np.sum(self.covMap[node[:-1]]) 
                    else:
                        newSum = self.directedUnitigBiGraph[innode][node]['covweight'] + self.lengths[node[:-1]]*self.KC[node[:-1]] 
                    self.directedUnitigBiGraph.add_edge(innode, node, weight=newWeight,covweight=newSum)
    
    
    def clearReadWeights(self):
    
        assert self.directedUnitigBiGraphS is not None
        
        for u,v in self.directedUnitigBiGraphS.edges():
            self.directedUnitigBiGraphS[u][v]['readweight'] = 0.    
        

    def setReadWeights(self, readGraphMaps, weights, ids):
    
        for i, id in enumerate(ids):
            map = readGraphMaps[id]
            
            for u, v in zip(map, map[1:]):
                self.directedUnitigBiGraphS[u][v]['readweight'] +=   weights[i]*self.directedUnitigBiGraphS[u][v]['weight']   
    
    
    def setDirectedBiGraphSource(self, sources,sinks):
    
        directedUnitigBiGraphS =  None
        
        tempDiGraph = self.directedUnitigBiGraph.copy()
        
        tempDiGraph.add_node('source+')
        
        tempDiGraph.add_node('sink+')
        
        for source in sources:
            tempDiGraph.add_edge('source+',source)
            
        for sink in sinks:
            tempDiGraph.add_edge(sink,'sink+')
        
        undirectedUnitigBiGraph = tempDiGraph.to_undirected()
        
        
        for c in nx.connected_components(undirectedUnitigBiGraph):

            if set(sources).issubset(set(c)) and set(sinks).issubset(set(c)):
                self.directedUnitigBiGraphS = nx.subgraph(tempDiGraph, c)
                 
                return True
    
        return False
        
    def getHeaviestBiGraphPath(self, value,sources,sinks):
        
        assert nx.is_directed_acyclic_graph(self.directedUnitigBiGraphS)

        dGraph = self.directedUnitigBiGraphS
        top_sort = list(nx.topological_sort(dGraph))
        lenSort = len(top_sort)
            
        maxPred = {}
        maxWeightNode = {}
        for node in top_sort:
            maxWeight = 0.
            maxPred[node] = None
            noded = node[:-1]
            
            if dGraph.out_degree(node) == 0:
                lengthPlus = self.lengths[noded]
            else:
                lengthPlus = self.lengths[noded] - self.overlapLength
            
            
            for predecessor in dGraph.predecessors(node):
                weight = maxWeightNode[predecessor] + dGraph[predecessor][node][value]*lengthPlus
                
                if weight > maxWeight:
                    maxWeight = weight
                    maxPred[node] = predecessor
            
            maxWeightNode[node] = maxWeight


        bestNode = None
        maxNodeWeight = 0.
        for node in top_sort:
            if maxWeightNode[node] > maxNodeWeight:
                maxNodeWeight = maxWeightNode[node] 
                bestNode = node
        
        minPath = []
        while bestNode is not None:
            minPath.append(bestNode)
            bestNode = maxPred[bestNode]
        minPath.reverse()
                            
        maxSeq = self.getUnitigWalk(minPath)
        

        return (minPath, maxSeq)
    
    def getHeaviestBiGraphPathNoLength(self, value,sources,sinks):
        
        assert nx.is_directed_acyclic_graph(self.directedUnitigBiGraphS)

        dGraph = self.directedUnitigBiGraphS
        top_sort = list(nx.topological_sort(dGraph))
        lenSort = len(top_sort)
            
        maxPred = {}
        maxWeightNode = {}
        for node in top_sort:
            maxWeight = 0.
            maxPred[node] = None
            noded = node[:-1]
            
            for predecessor in dGraph.predecessors(node):
                weight = maxWeightNode[predecessor] + dGraph[predecessor][node][value]
                
                if weight > maxWeight:
                    maxWeight = weight
                    maxPred[node] = predecessor
            
            maxWeightNode[node] = maxWeight


        bestNode = None
        maxNodeWeight = 0.
        for node in top_sort:
            if maxWeightNode[node] > maxNodeWeight:
                maxNodeWeight = maxWeightNode[node] 
                bestNode = node
        
        minPath = []
        while bestNode is not None:
            minPath.append(bestNode)
            bestNode = maxPred[bestNode]
        minPath.reverse()
                            
        maxSeq = self.getUnitigWalk(minPath)
        

        return (minPath, maxSeq)
        
    
    def getHeaviestBiGraphPathUnitigNode(self,unitigValueDir,sources,sinks):
        
        assert self.directedUnitigBiGraphS is not None

        dGraph = self.directedUnitigBiGraphS
        top_sort = list(nx.topological_sort(dGraph))
        lenSort = len(top_sort)
            
        maxPred = {}
        maxWeightNode = {}
        for node in top_sort:
        
            pred = list(dGraph.predecessors(node))
            
            if len(pred) > 0:
                maxWeightPred = maxWeightNode[pred[0]]
                maxPred[node] = pred[0]
                
                        
                for predecessor in pred[1:]:
                    weight = maxWeightNode[predecessor] 
                    
                    if weight > maxWeightPred:
                        maxWeightPred = weight
                        maxPred[node] = predecessor
            
                noded = node[:-1]
                
                maxWeightNode[node]  = maxWeightPred + unitigValueDir[noded]
            else:
                maxWeightNode[node] = 0.
                maxPred[node] = None
            

       # max_key = max(a_dictionary, key=a_dictionary.get)

        #bestNode = None
        #maxNodeWeight = 0.
        #for node in top_sort:
         #   if maxWeightNode[node] > maxNodeWeight:
          #      maxNodeWeight = maxWeightNode[node] 
           #     bestNode = node
        
        minPath = []
        bestNode = 'sink+'
        while bestNode is not None:
            minPath.append(bestNode)
            bestNode = maxPred[bestNode]
        
        minPath.pop(0)
        minPath.pop()
        minPath.reverse()
                            
        maxSeq = self.getUnitigWalk(minPath)
        

        return (minPath, maxSeq)
    
    
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

    def propagateEndPath(self, path, endPos):
    
        endUnitig = path[-1][0]
        endNodeName = convertNodeToName(path[-1])
        
        
        if endPos > self.lengths[endUnitig] - self.overlapLength:
            
            out_nodes = list(self.directedUnitigBiGraph.successors(endNodeName))
            
            if len(out_nodes) == 1:
                newEndPos = endPos - self.lengths[endUnitig] + self.overlapLength
                
                path.append(convertNameToNode2(out_nodes[0]))
                
                endPos = newEndPos
                self.propagateEndPath(path, endPos)
        return endPos
    
    def computeMeanCoverage(self, length):
        
        if self.covMap is None:
            raise ValueError()
        
        if len(self.unitigs) > 0:
            covMean = np.zeros(self.covMap[self.unitigs[0]].shape)
            lengthSum = 0.
            
            for unitig in self.unitigs:
                covMean += self.lengths[unitig]*self.covMap[unitig]
                lengthSum += self.lengths[unitig]
            
            return covMean/length
        
        else:
            return None
            
    def selectSamples(self, selectedSamples):
    
        if self.covMap is None:
            raise ValueError()
        
        for unitig in self.unitigs:
            self.covMap[unitig] = self.covMap[unitig][selectedSamples]
            

    def computePathCoverage(self, path):
        
        if self.covMap is None:
            raise ValueError()
        
        if len(path) > 0:
            totalLength = 0.
       
            covSum = np.zeros_like(self.covMap[self.unitigs[0]])
        
            for noded in path[:-1]:
                node = noded[:-1]
                lengthPlus = self.lengths[node] - self.overlapLength
                totalLength += lengthPlus
                covSum += lengthPlus*self.covMap[node]
            nodeLast = path[-1][:-1]
            totalLength += self.lengths[nodeLast]
            covSum += self.lengths[nodeLast]*self.covMap[nodeLast]
        
            return covSum/totalLength
        else:
            covSum = np.zeros_like(self.covMap[self.unitigs[0]])
            return covSum    
    
    def propagateStartPath(self, path, startPos):
    
        startUnitig = path[0][0]
        startNodeName = convertNodeToName(path[0])
        
        if startPos < self.overlapLength:
            
            in_nodes = list(self.directedUnitigBiGraph.predecessors(startNodeName))
            
            if len(in_nodes) == 1:
                inNodeCode = convertNameToNode2(in_nodes[0])
                
                newStartPos = startPos + self.lengths[inNodeCode[0]] - self.overlapLength
                
                path.insert(0,inNodeCode)
                
                startPos = newStartPos
                
                self.propagateStartPath(path, startPos)
        return startPos
        
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
    
        
        
