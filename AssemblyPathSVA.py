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
import math
from subprocess import Popen, PIPE, STDOUT
from operator import mul, truediv, eq, ne, add, ge, le, itemgetter
import networkx as nx
import argparse

import collections
from collections import deque
from collections import defaultdict
from collections import Counter
from numpy.random import RandomState

from graph import Graph
from Utils import convertNodeToName
from Utils import read_unitig_order_file
from Utils import elop
from Utils import expNormLogProb
from UnitigGraph import UnitigGraph
from NMF import NMF

class AssemblyPathSVA():
    """ Class for structured variational approximation on Assembly Graph"""    
    minW = 1.0e-3    
    def __init__(self, prng, assemblyGraphs, source_maps, sink_maps, G = 2, maxFlux=2, readLength = 100, epsilon = 1.0e-5):
        self.prng = prng #random state to store

        self.readLength = readLength #sequencing read length
 
        self.assemblyGraphs = assemblyGraphs
        
        self.source_maps = source_maps
        
        self.sink_maps = sink_maps
 
        self.factorGraphs = {} # dict of factorGraphs as pyfac Graphs
        self.unitigFactorNodes = {}
        self.maxFlux = 2
        
        #define dummy source and sink node names
        self.sinkNode = 'sink+'
        
        self.sourceNode = 'source+'
        
        self.V = 0
        self.mapIdx = {}
        self.adjLengths = {}
        self.covMapAdj = {}
        self.unitigs = []
        self.mapGeneIdx = collections.defaultdict(dict)
        bFirst = True
        for gene, assemblyGraph in assemblyGraphs.items():
            
            (factorGraph, unitigFactorNode) = self.createFactorGraph(assemblyGraph, source_maps[gene], sink_maps[gene])
            
            self.factorGraphs[gene] = factorGraph
            self.unitigFactorNodes[gene] = unitigFactorNode
            
            unitigAdj = [gene + "_" + s for s in assemblyGraph.unitigs]
            self.unitigs.extend(unitigAdj)
            for (unitigNew, unitig) in zip(unitigAdj,assemblyGraph.unitigs):
                self.adjLengths[unitigNew] = assemblyGraph.lengths[unitig] - 2.0*assemblyGraph.overlapLength + 2.0*self.readLength
                assert self.adjLengths[unitigNew] > 0
                self.mapIdx[unitigNew] = self.V
                self.mapGeneIdx[gene][unitig] = self.V 
                self.covMapAdj[unitigNew] = (assemblyGraph.covMap[unitig] * self.adjLengths[unitigNew])/self.readLength
                
                if bFirst:
                    self.S = assemblyGraph.covMap[unitig].shape[0]
                    bFirst = False
                
                self.V += 1
                
        self.X = np.zeros((self.V,self.S))
        self.XN = np.zeros((self.V,self.S))
        self.lengths = np.zeros(self.V)
        
        idx = 0
        for v in self.unitigs:
            covName = None
            if v in self.covMapAdj:
                covName = self.covMapAdj[v]
            else:
                print("Serious problem")
                
            self.lengths[idx] = self.adjLengths[v]
            self.X[idx,:] = covName
            self.XN[idx,:] = covName/self.lengths[idx] 
            idx=idx+1
       
        self.XD = np.floor(self.X).astype(int)
        
        #Now initialise SVA parameters
        self.G = G
        
        #list of mean assignments of strains to graph
        self.phiMean = np.zeros((self.V,self.G))
        self.phiMean2 = np.zeros((self.V,self.G))
        
        self.epsilon = epsilon #parameter for gamma exponential prior
        self.muGamma = np.zeros((self.G,self.S))
        self.muGamma2 = np.zeros((self.G,self.S))
        self.tauGamma = np.zeros((self.G,self.S))
        
        #current excitations on the graph
        self.eLambda = np.zeros((self.V,self.S))
        
        self.margG = [None]*self.G
        
        self.elbo = 0.
        
        self.tau = 1.0
        
    def writeNetworkGraph(self,networkGraph, fileName):
        copyGraph = networkGraph.copy()
        
        for (n,d) in copyGraph.nodes(data=True):
            del d["code"]
        
        nx.write_graphml(copyGraph,fileName)
        
    def createFactorGraph(self, assemblyGraph, sources, sinks):
    
        tempGraph = self.createTempGraph(assemblyGraph, assemblyGraph.unitigs)
    
        self.addSourceSink(tempGraph, sources, sinks)
        
        #use largest connected factor graph
        tempUFactor = tempGraph.to_undirected()
        
        compFactor = sorted(nx.connected_components(tempUFactor),key = len, reverse=True)
        fNodes = list(tempGraph.nodes())
        if len(compFactor) > 1:
            largestFactor = compFactor[0]
            
            for node in fNodes:
                if node not in largestFactor:
                    tempGraph.remove_node(node)
        
        self.writeNetworkGraph(tempGraph,"temp.graphml")
    
        (factorGraph, unitigFactorNodes) = self.generateFactorGraph(tempGraph, assemblyGraph.unitigs)
    
        return (factorGraph, unitigFactorNodes)
    
    def generateFactorGraph(self, factorGraph, unitigs):
        probGraph = Graph()
        unitigFactorNodes = {}
        
        for node in factorGraph:            
            if 'factor' not in factorGraph.node[node]:
                print(str(node))
            else:
                if not factorGraph.node[node]['factor']:
                #just add edge as variable
                    probGraph.addVarNode(node,self.maxFlux)
            

        for node in factorGraph:
            
            if factorGraph.node[node]['factor']:
                inNodes = list(factorGraph.predecessors(node))
                
                outNodes = list(factorGraph.successors(node))
                
                nIn = len(inNodes)
                
                nOut = len(outNodes)
                
                Ntotal = nIn + nOut
                
                factorMatrix = np.zeros([self.maxFlux]*Ntotal)
                
                for indices, value in np.ndenumerate(factorMatrix):
                
                    fIn = sum(indices[0:nIn])
                    fOut = sum(indices[nIn:])
                    
                    if fIn == fOut:
                        factorMatrix[indices] = 1.0
                        
                mapNodeList = [probGraph.mapNodes[x] for x in inNodes + outNodes]
                        
                #mapNodes[node] = probGraph.addFacNode(factorMatrix, *mapNodeList)
                probGraph.addFacNode(factorMatrix, *mapNodeList)
    
        unitigFacNodes = {}
        
        for unitig in unitigs:
            
            plusNode = unitig + "+"
            inNodesPlus = []
            if plusNode in factorGraph:
                inNodesPlus = list(factorGraph.predecessors(plusNode))
            
            minusNode = unitig + "-"
            inNodesMinus = []
            if minusNode in factorGraph:
                inNodesMinus = list(factorGraph.predecessors(minusNode))
            
            nInPlus = len(inNodesPlus)
                
            nInMinus = len(inNodesMinus)
            
            Ntotal = nInPlus + nInMinus
            
            if Ntotal > 0:
                mapNodesF = [probGraph.mapNodes[x] for x in inNodesPlus + inNodesMinus]
                nMax = Ntotal*(self.maxFlux - 1) + 1
                probGraph.addVarNode(unitig,nMax)
           
                dimF = [nMax] + [self.maxFlux]*Ntotal
           
                fluxMatrix = np.zeros(dimF)
                dummyMatrix = np.zeros([self.maxFlux]*Ntotal)
      
                for indices, value in np.ndenumerate(dummyMatrix):
            
                    tIn = sum(indices)
                
                    fluxMatrix[tuple([tIn]+ list(indices))] = 1.0
                
                
                probGraph.addFacNode(fluxMatrix, *([probGraph.mapNodes[unitig]] + mapNodesF))
            
                discreteMatrix = np.zeros((nMax,1))
            
                unitigFacNodes[unitig] = probGraph.addFacNode(discreteMatrix, probGraph.mapNodes[unitig])
        
        return (probGraph,unitigFacNodes)
    
    
    def createTempGraph(self, assemblyGraph, unitigs):
        
        factorGraph = nx.DiGraph()
        
        for node in unitigs:
    
            #add positive and negative version of untig to factor graph
            nodePlus  = (node, True)
            nodeMinus = (node, False)
            
            nodePlusName = convertNodeToName(nodePlus)
            nodeMinusName = convertNodeToName(nodeMinus)
            
            if nodePlusName not in factorGraph:
                factorGraph.add_node(nodePlusName, factor=True, code=nodePlus)
            
            if nodeMinusName not in factorGraph:
                factorGraph.add_node(nodeMinusName, factor=True, code=nodeMinus)
        
            #get all links outgoing given +ve direction on node
            for outnode, dirns in assemblyGraph.overlaps[node].items():

                for dirn in dirns:
                    (start,end) = dirn
            
                    #add outgoing positive edge
                
                    if start:
                        addEdge = (nodePlus, (outnode, end)) 
    
                    else:
                        #reverse as incoming edge
                        addEdge = ((outnode,not end), (node, True))
                
                    edgeName = convertNodeToName(addEdge)
                    if edgeName not in factorGraph:
                        factorGraph.add_node(edgeName, factor=False, code=addEdge)
                    
                    if start:
                        factorGraph.add_edge(nodePlusName, edgeName)
                    else:
                        factorGraph.add_edge(edgeName, nodePlusName)        
            
                    #add negative edges
                    if start:
                        #reverse as incoming edge
                        addEdge = ((outnode, not end),(nodeMinus))
                    else:    
                        addEdge = (nodeMinus, (outnode, end)) 
                
                    edgeName = convertNodeToName(addEdge)
                    if edgeName not in factorGraph:
                        factorGraph.add_node(edgeName, factor=False, code=addEdge)
                
                    if start:
                        factorGraph.add_edge(edgeName, nodeMinusName)
                    else:
                        factorGraph.add_edge(nodeMinusName, edgeName)
            
        return factorGraph
            
    
    def addSourceSink(self, tempGraph, sources, sinks):
    
        #add sink factor and dummy flow out

        if self.sinkNode not in tempGraph:
            tempGraph.add_node(self.sinkNode, factor=True, code=('sink',True))
        
        node_code = (('sink',True),('infty',True))
        
        sinkEdge = convertNodeToName(node_code)
        
        tempGraph.add_node(sinkEdge, factor=False, code=node_code)
        tempGraph.add_edge(self.sinkNode, sinkEdge)
        
        if self.sourceNode not in tempGraph:
            tempGraph.add_node(self.sourceNode, factor=True, code=('source',True))
        
        node_code = (('zero',True),('source',True))
        sourceEdge = convertNodeToName(node_code)
        tempGraph.add_node(sourceEdge, factor=False, code=node_code)
        tempGraph.add_edge(sourceEdge, self.sourceNode)
        
        for (sinkid,dirn) in sinks:
            
            node_code = ((sinkid,dirn),('sink',True))
            
            edgeName = convertNodeToName(node_code)
            
            sinkName = convertNodeToName((sinkid,dirn))
            
            tempGraph.add_node(edgeName, factor=False, code=node_code)
            
            tempGraph.add_edge(sinkName,edgeName)
            
            tempGraph.add_edge(edgeName,self.sinkNode)
        
        for (sourceid,dirn) in sources:
            
            node_code = (('source',True),(sourceid,dirn))
            
            edgeName = convertNodeToName(node_code)
            
            sourceName = convertNodeToName((sourceid,dirn))
            
            tempGraph.add_node(edgeName, factor=False, code=node_code)
            
            tempGraph.add_edge(edgeName,sourceName)
            
            tempGraph.add_edge(self.sourceNode,edgeName)
   
    def parseMargString(self, factorGraph, outputString):
        mapMarg = {}
        lines = outputString.split('\\n')
        #({x410}, (0.625, 0.375, 0))
         
        for line in lines:
            matchP = re.search(r'\((.*)\)',line)
            
            if matchP is not None:
                matchedP = matchP.group(1)
                
                matchVar = re.search(r'\{x(.*?)\}',matchedP)
                
                if matchVar is not None:
                    var = matchVar.group(1)
                    
                    matchVals = re.search(r'\((.*?)\)',matchedP)
                    
                    if matchVals is not None:
                        vals = matchVals.group(1).split(',')
                        floatVals = [float(i) for i in vals]
                        if int(var) >= len(factorGraph.varNames):
                            varName = "Debug"
                        else:
                            varName = factorGraph.varNames[int(var)]
                        if varName is not None:                
                            mapMarg[varName] = np.asarray(floatVals)
        return mapMarg
    
    def updateUnitigFactors(self, unitigs, unitigMap, unitigFacNodes, gidx, tau):
        
        mapGammaG = self.muGamma[gidx,:]
        mapGammaG2 = self.muGamma2[gidx,:]
        dSum2 = np.sum(mapGammaG2)
        
        for unitig in unitigs:
            if unitig in unitigFacNodes:
                unitigFacNode = unitigFacNodes[unitig]
                v_idx = unitigMap[unitig]
                P = unitigFacNode.P
                tempMatrix = np.zeros_like(P)
            
                currELambda = self.eLambda[v_idx,:]
                
                lengthNode = self.lengths[v_idx]
            
                nMax = unitigFacNode.P.shape[0]
                
                dFac = -0.5*tau*lengthNode
                
                T1 = mapGammaG*(lengthNode*currELambda - self.X[v_idx,:])
                dVal1 = 2.0*np.sum(T1)
                dVal2 = lengthNode*dSum2
                
                for d in range(nMax):
                    tempMatrix[d] = dFac*(dVal1*float(d) + dVal2*float(d*d))    

                unitigFacNode.P = expNormLogProb(tempMatrix)

    def updateUnitigFactorsW(self,unitigs, unitigMap, unitigFacNodes, W,gidx):

        for unitig in unitigs:
            if unitig in unitigFacNodes:
                unitigFacNode = unitigFacNodes[unitig]
                v_idx = unitigMap[unitig]
                P = unitigFacNode.P
                tempMatrix = np.zeros_like(P)
                w = W[v_idx,gidx]

                wmax = P.shape[0] - 1
                tempMatrix.fill(self.minW) 

                if w > wmax:
                    w = wmax
                
                iw = int(w)
                difw = w - iw
                tempMatrix[iw] = 1.0 - difw
                
                if iw < wmax:
                    tempMatrix[iw + 1] = difw

                unitigFacNode.P = tempMatrix


    def removeGamma(self,g_idx):
        
        meanAss = self.phiMean[:,g_idx]
        gammaG  = self.muGamma[g_idx,:]
        
        self.eLambda -= meanAss[:,np.newaxis]*gammaG[np.newaxis,:]

    def addGamma(self,g_idx):
        
        meanAss = self.phiMean[:,g_idx]
        gammaG  = self.muGamma[g_idx,:]
        
        self.eLambda += meanAss[:,np.newaxis]*gammaG[np.newaxis,:]

    def updateGamma(self,g_idx):
        
        temp = np.delete(self.muGamma,g_idx,,0)
        temp2 = np.delete(self.phiMean,g_idx,1)
        
        numer = self.phiMean[:,g_idx]*(self.X - np.dot(temp2,temp)*self.lengths)
        
        denom = self.lengths*self.phiMean2[:,g_idx]
        
        newGamma = np.sum(numer,0)/np.sum(denom)

        return newGamma

    def updatePhiMean(self,unitigs,mapUnitig,marg,g_idx):
    
        for unitig in unitigs:
        
            if unitig in marg:
                v_idx = mapUnitig[unitig]
                ND = marg[unitig].shape[0]
                self.phiMean[v_idx,g_idx] = np.sum(marg[unitig]*np.arange(ND))
                d2 = np.square(np.arange(ND))
                self.phiMean2[v_idx,g_idx] = np.sum(marg[unitig]*d2)
                
    def update(self, maxIter):
    
        iter = 0
    
        while iter < maxIter:
            #update phi marginals
            
            for g in range(self.G):
                
                self.removeGamma(g)
            
                for gene, factorGraph in self.factorGraphs.items():
                    unitigs = self.assemblyGraphs[gene].unitigs
                    
                    if iter < 10:
                        self.tau = 0.01                    
                    else:
                        self.tau = 1.

                    self.updateUnitigFactors(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], g, self.tau)
                    
        
                    factorGraph.reset()
        
                    factorGraph.var['zero+source+'].condition(1)

                    factorGraph.var['sink+infty+'].condition(1)
                    
                    graphString = str(factorGraph)
                    graphFileName = 'graph_'+ str(g) + '.fg'
                
                    with open(graphFileName, "w") as text_file:
                        print(graphString, file=text_file)
                
                    cmd = './runfg_marg ' + graphFileName + ' 0'
                
                    p = Popen(cmd, stdout=PIPE,shell=True)
        
                    outString = p.stdout.read()
               
                    margP = self.parseMargString(factorGraph,str(outString))
                    if len(margP) > 0: 
                        self.margG[g] = self.parseMargString(factorGraph,str(outString))
       
                    self.updatePhiMean(unitigs,self.mapGeneIdx[gene],self.margG[g],g)
       
                self.addGamma(g)
            
            for g in range(self.G):
                newGammaG = self.updateGamma(g)
               
            print(str(iter)+","+ str(self.divF()))  
            iter += 1
    
    def div(self):
        """Compute divergence of target matrix from its NMF estimate."""
        Va = self.eLambda
        return (np.multiply(self.XN, np.log(elop(self.XN, Va, truediv))) - self.XN + Va).sum()

    def divF(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""
        R = self.eLambda - self.XN
        return np.multiply(R, R).sum()

    def initNMF(self):
        
        covNMF =  NMF(self.XN,self.G,n_run = 10)
    
        covNMF.factorize()
        covNMF.factorizeH()

        self.muGamma = np.copy(covNMF.H)
        covNMF.factorizeW()
        
        initEta = covNMF.W
            
        for g in range(self.G):
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.assemblyGraphs[gene].unitigs
                    
                self.updateUnitigFactorsW(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], initEta, g)
                  
                factorGraph.reset()
        
                factorGraph.var['zero+source+'].condition(1)

                factorGraph.var['sink+infty+'].condition(1)
                    
                graphString = str(factorGraph)
                graphFileName = 'graph_'+ str(g) + '.fg'
                
                with open(graphFileName, "w") as text_file:
                    print(graphString, file=text_file)
                
                cmd = './runfg_marg ' + graphFileName + ' 0'
                
                p = Popen(cmd, stdout=PIPE,shell=True)
                
                outString = p.stdout.read()
                
                self.margG[g] = self.parseMargString(factorGraph,str(outString))
                self.updatePhiMean(unitigs,self.mapGeneIdx[gene],self.margG[g],g)
            self.addGamma(g)    
        print("-1,"+ str(self.div())) 

    def initNMFGamma(self,gamma):
        
        covNMF =  NMF(self.XN,self.G,n_run = 10)
        covNMF.random_initialize() 
        covNMF.H = np.copy(gamma)
        covNMF.factorizeW()
        
        initEta = covNMF.W
            
        for g in range(self.G):
            
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.assemblyGraphs[gene].unitigs
                    
                self.updateUnitigFactorsW(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], initEta, g)
                  
                factorGraph.reset()
        
                factorGraph.var['zero+source+'].condition(1)

                factorGraph.var['sink+infty+'].condition(1)
                    
                graphString = str(factorGraph)
                graphFileName = 'graph_'+ str(g) + '.fg'
                
                with open(graphFileName, "w") as text_file:
                    print(graphString, file=text_file)
                
                cmd = './runfg_marg ' + graphFileName + ' 0'
                
                p = Popen(cmd, stdout=PIPE,shell=True)
                
                outString = p.stdout.read()
                
                self.margG[g] = self.parseMargString(factorGraph,str(outString))
            
                self.updatePhiMean(unitigs,self.mapGeneIdx[gene],self.margG[g],g)
            self.addGamma(g)    
        print("-1,"+ str(self.div())) 

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("unitig_file", help="unitig fasta file in Bcalm2 format")

    parser.add_argument("cov_file", help="coverage file")

    parser.add_argument("kmer_length", help="kmer length assumed overlap")

    parser.add_argument("unitig_order_file", help="csv node file")

    parser.add_argument("gamma_file", help="csv node file")

    parser.add_argument('-g','--strain_number',nargs='?', default=5, type=int, 
        help=("maximum number of strains"))

    parser.add_argument('-f','--frac',nargs='?', default=0.75, type=float, 
        help=("fraction for path source sink"))

    args = parser.parse_args()

    import ipdb; ipdb.set_trace()

    np.random.seed(2)
    prng = RandomState(238329)
                
    unitig_order = read_unitig_order_file(args.unitig_order_file)
                
    unitigGraph = UnitigGraph.loadGraph(args.unitig_file, 71, args.cov_file)   
  
    #get separate components in graph
    components = sorted(nx.connected_components(unitigGraph.undirectedUnitigGraph), key = len, reverse=True)
    assemblyGraphs = {}
    sink_maps = {}
    source_maps = {}
    c = 0
    for component in components:
        if c == 0:
            unitigSubGraph = unitigGraph.createUndirectedGraphSubset(component)
            assemblyGraphs[str(c)] = unitigSubGraph
            
            sub_unitig_order = {k: unitig_order[k] for k in component}
        
            (source_list, sink_list) = unitigSubGraph.selectSourceSinks2(args.frac)

            source_names = [convertNodeToName(source) for source in source_list] 
            sink_names = [convertNodeToName(sink) for sink in sink_list]
            
            sink_maps[str(c)] = sink_list
            source_maps[str(c)] = source_list
        c = c + 1
    assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G = args.strain_number, readLength=150)
    
    covs    = p.read_csv(args.gamma_file, header=0, index_col=0)
    
    covs.drop('Strain', axis=1, inplace=True)
    
    cov_matrix = covs.values
    
    assGraph.muGamma = cov_matrix/assGraph.readLength
    assGraph.muGamma2 = np.square(assGraph.muGamma)
    assGraph.initNMFGamma(assGraph.muGamma)
    #assGraph.tau = 1.0e-4
    assGraph.update(100)
    
if __name__ == "__main__":
    main(sys.argv[1:])
