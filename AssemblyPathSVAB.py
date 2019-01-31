import uuid
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
from scipy.special import psi as digamma

from copy import deepcopy
from copy import copy

import math
import subprocess
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
from Utils.UtilsFunctions import convertNodeToName
from Utils.UtilsFunctions import elop
from Utils.UtilsFunctions import expNormLogProb
from Utils.UtilsFunctions import TN_vector_expectation
from Utils.UtilsFunctions import TN_vector_variance
from Utils.UtilsFunctions import readRefAssign
from Utils.UnitigGraph import UnitigGraph
from NMFM import NMF

from mask import compute_folds
 
import multiprocessing
import subprocess
import shlex

from multiprocessing.pool import ThreadPool

def call_proc(cmd):
    """ This runs in a separate thread. """
    #subprocess.call(shlex.split(cmd))  # This will block until cmd finishes
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)


class AssemblyPathSVA():
    """ Class for structured variational approximation on Assembly Graph"""    
    minW = 1.0e-3    
    def __init__(self, prng, assemblyGraphs, source_maps, sink_maps, G = 2, maxFlux=2, 
                readLength = 100, epsilon = 1.0e5,alpha=0.01,beta=0.01,alpha0=1.0e-9,beta0=1.0e-9,
                no_folds = 10, ARD = False, BIAS = True, muTheta0 = 1.0, tauTheta0 = 10.0):
        self.prng = prng #random state to store

        self.readLength = readLength #sequencing read length
 
        self.assemblyGraphs = assemblyGraphs
        
        self.source_maps = source_maps
        
        self.sink_maps = sink_maps
 
        self.factorGraphs = {} # dict of factorGraphs as pyfac Graphs

        self.factorDiGraphs = {} # dict of factorGraphs as networkx diGraphs

        self.unitigFactorNodes = {}
        self.maxFlux = 2
        
        #define dummy source and sink node names
        self.sinkNode = 'sink+'
        
        self.sourceNode = 'source+'
        
        self.no_folds = no_folds
        
        #prior parameters for Gamma tau
        self.alpha = alpha
        self.beta  = beta
        self.ARD = ARD
        if self.ARD:
            self.alpha0, self.beta0 = alpha0, beta0
            
        self.BIAS = BIAS
        if self.BIAS:
            self.muTheta0 = muTheta0
            self.tauTheta0 = tauTheta0

        self.V = 0
        self.mapIdx = {}
        self.mapUnitigs = {}
        self.adjLengths = {}
        self.covMapAdj = {}
        
        self.unitigs = []
        self.genes = []
        self.mapGeneIdx = collections.defaultdict(dict)
        bFirst = True
        for gene, assemblyGraph in assemblyGraphs.items():
            self.genes.append(gene)
            
            (factorGraph, unitigFactorNode, factorDiGraph) = self.createFactorGraph(assemblyGraph, source_maps[gene], sink_maps[gene])
           
            unitigsDash = list(unitigFactorNode.keys())
            unitigsDash.sort() 
            self.factorGraphs[gene] = factorGraph

            self.factorDiGraphs[gene] = factorDiGraph 

            self.unitigFactorNodes[gene] = unitigFactorNode
            #unitigList = list(assemblyGraph.unitigs)
            #unitigList.sort(key=int)
            self.mapUnitigs[gene] = unitigsDash
            unitigAdj = [gene + "_" + s for s in unitigsDash]
            self.unitigs.extend(unitigAdj)
            for (unitigNew, unitig) in zip(unitigAdj,unitigsDash):
                if unitig.startswith('connect'):
                    self.adjLengths[unitigNew] = 0.0
                else:
                    self.adjLengths[unitigNew] = assemblyGraph.lengths[unitig] - 2.0*assemblyGraph.overlapLength + 2.0*self.readLength
                #self.adjLengths[unitigNew] = assemblyGraph.lengths[unitig] - assemblyGraph.overlapLength + self.readLength
                    assert self.adjLengths[unitigNew] > 0
                
                self.mapIdx[unitigNew] = self.V
                self.mapGeneIdx[gene][unitig] = self.V 
                self.covMapAdj[unitigNew] = (assemblyGraph.covMap[unitig] * float(self.adjLengths[unitigNew]))/self.readLength
                
                if bFirst:
                    self.S = assemblyGraph.covMap[unitig].shape[0]
                    bFirst = False
                
                self.V += 1
                
        self.X = np.zeros((self.V,self.S))
        self.XN = np.zeros((self.V,self.S))
        self.lengths = np.zeros(self.V)
        
        #note really need to remove unreachable nodes from calculations
        self.logPhiPrior = np.zeros(self.V)
        for gene, unitigFactorNode in self.unitigFactorNodes.items(): 
            self.setPhiConstant(self.mapUnitigs[gene], self.mapGeneIdx[gene], unitigFactorNode)

 
        idx = 0
        for v in self.unitigs:
            covName = None
            if v in self.covMapAdj:
                covName = self.covMapAdj[v]
            else:
                print("Serious problem")
                
            self.lengths[idx] = self.adjLengths[v]
            self.X[idx,:] = covName
            if self.lengths[idx] > 0.:
                self.XN[idx,:] = covName/self.lengths[idx] 
            
            idx=idx+1
       
        self.XD = np.floor(self.X).astype(int)
        
        #create mask matrices
        (self.M_trains, self.M_tests) = compute_folds(self.V,self.S,self.no_folds)
        self.M_train = self.M_trains[0]
        self.M_train = np.ones((self.V,self.S))
        self.M_test = self.M_tests[0]
        self.m = 0
        #Now initialise SVA parameters
        self.G = G
        self.Omega = self.M_train.sum()      
 
        #list of mean assignments of strains to graph
        self.expPhi = np.zeros((self.V,self.G))
        self.expPhi2 = np.zeros((self.V,self.G))
        self.HPhi = np.zeros((self.V,self.G))

        self.epsilon = epsilon #parameter for gamma exponential prior
        self.expGamma = np.zeros((self.G,self.S)) #expectation of gamma
        self.expGamma2 = np.zeros((self.G,self.S))
        
        self.muGamma = np.zeros((self.G,self.S))
        self.tauGamma = np.zeros((self.G,self.S))
        self.varGamma = np.zeros((self.G,self.S))
        #current excitations on the graph
        self.eLambda = np.zeros((self.V,self.S))
        
        self.margG = dict()
        for gene in self.genes:
            self.margG[gene] = [dict() for x in range(self.G)]

        if self.ARD:
            self.alphak_s, self.betak_s = np.zeros(self.G), np.zeros(self.G)
            self.exp_lambdak, self.exp_loglambdak = np.zeros(self.G), np.zeros(self.G)
            for g in range(self.G):
                self.alphak_s[g] = self.alpha0
                self.betak_s[g] = self.beta0
                self.update_exp_lambdak(g)
        
        if self.BIAS:
            self.expTheta  = np.ones(self.V)
            self.expTheta.fill(self.muTheta0)
            
            self.expTheta2 = np.ones(self.V)
            self.expTheta2.fill(self.muTheta0*self.muTheta0)
            
            self.muTheta = np.ones(self.V)
            self.muTheta.fill(self.muTheta0)
            
            self.tauTheta = np.ones(self.V)
            self.tauTheta.fill(self.tauTheta0)

            self.varTheta = 1.0/self.tauTheta 
            
        self.elbo = 0.
        self.expTau = 1.0
        self.alphaTau = 1.0
        self.betaTau = 1.0
    
    @classmethod
    def copyGamma(cls,assGraphG, assGraphH):
    
        copyGraphG = copy(assGraphG)
    
        copyGraphG.G = assGraphH.G
 
        #list of mean assignments of strains to graph
        copyGraphG.expPhi = np.zeros((copyGraphG.V,copyGraphG.G))
        copyGraphG.expPhi2 = np.zeros((copyGraphG.V,copyGraphG.G))
        copyGraphG.HPhi = np.zeros((copyGraphG.V,copyGraphG.G))

        copyGraphG.epsilon = assGraphH.epsilon #parameter for gamma exponential prior
        
        copyGraphG.expGamma = np.copy(assGraphH.expGamma)
    
        copyGraphG.expGamma2 = np.copy(assGraphH.expGamma2)
        
        copyGraphG.muGamma = np.copy(assGraphH.muGamma)
        copyGraphG.tauGamma = np.copy(assGraphH.tauGamma)
        copyGraphG.varGamma = np.copy(assGraphH.varGamma)
        #current excitations on the graph
        copyGraphG.eLambda = np.zeros((copyGraphG.V,copyGraphG.S))
        
        copyGraphG.margG = dict()
        for gene in copyGraphG.genes:
            copyGraphG.margG[gene] = [dict() for x in range(copyGraphG.G)]

        if copyGraphG.ARD:
            copyGraphG.alphak_s = np.copy(assGraphH.alphak_s)
            copyGraphG.betak_s = np.copy(assGraphH.betak_s)
            copyGraphG.exp_lambdak =  np.copy(assGraphH.exp_lambdak)
            copyGraphG.exp_loglambdak = np.copy(assGraphH.exp_loglambdak)
        
        if copyGraphG.BIAS:
            copyGraphG.expTheta   = np.copy(assGraphH.expTheta)
            copyGraphG.expTheta2  = np.copy(assGraphH.expTheta2)
            copyGraphG.muTheta    = np.copy(assGraphH.muTheta)
            copyGraphG.tauTheta   = np.copy(assGraphH.tauTheta)
            copyGraphG.varTheta   = np.copy(assGraphH.varTheta)
        
        copyGraphG.elbo = 0.
        copyGraphG.expTau = assGraphH.expTau
        copyGraphG.alphaTau = assGraphH.alphaTau
        copyGraphG.betaTau = assGraphH.betaTau
        
        return copyGraphG
    
    def addGenes(self,assemblyToAdd):
    
        self.assemblyGraphs.update(assemblyToAdd.assemblyGraphs)
        
        self.source_maps.update(assemblyToAdd.source_maps)
        
        self.sink_maps.update(assemblyToAdd.sink_maps)
 
        self.factorGraphs.update(assemblyToAdd.factorGraphs) 

        self.factorDiGraphs.update(assemblyToAdd.factorDiGraphs)

        self.unitigFactorNodes.update(assemblyToAdd.unitigFactorNodes)

        self.V += assemblyToAdd.V
                
        self.mapUnitigs.update(assemblyToAdd.mapUnitigs)  
        self.adjLengths.update(assemblyToAdd.adjLengths) 
        self.covMapAdj.update(assemblyToAdd.covMapAdj)
        
        self.unitigs = []
        self.genes = []
        self.mapIdx = {}
        self.mapGeneIdx = collections.defaultdict(dict)
        
        bFirst = True
        
        self.V = 0
        for gene, assemblyGraph in self.assemblyGraphs.items():
            self.genes.append(gene)
            unitigFactorNode = self.unitigFactorNodes[gene]
    
            unitigsDash = list(unitigFactorNode.keys())
            unitigsDash.sort(key=int) 
    
            unitigAdj = [gene + "_" + s for s in unitigsDash]
            self.unitigs.extend(unitigAdj)
            
            for (unitigNew, unitig) in zip(unitigAdj,unitigsDash):
            
                self.mapIdx[unitigNew] = self.V
                self.mapGeneIdx[gene][unitig] = self.V 
                
                self.V += 1
                
        self.X = np.zeros((self.V,self.S))
        self.XN = np.zeros((self.V,self.S))
        self.lengths = np.zeros(self.V)
        
        #note really need to remove unreachable nodes from calculations
        self.logPhiPrior = np.zeros(self.V)
        for gene, unitigFactorNode in self.unitigFactorNodes.items(): 
            self.setPhiConstant(self.mapUnitigs[gene], self.mapGeneIdx[gene], unitigFactorNode)

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
        
        #create mask matrices
        (self.M_trains, self.M_tests) = compute_folds(self.V,self.S,self.no_folds)
        self.M_train = self.M_trains[0]
        self.M_train = np.ones((self.V,self.S))
        self.M_test = self.M_tests[0]
        self.m = 0

        self.Omega = self.M_train.sum()      
 
        #list of mean assignments of strains to graph
        self.expPhi = np.zeros((self.V,self.G))
        self.expPhi2 = np.zeros((self.V,self.G))
        self.HPhi = np.zeros((self.V,self.G))

        #current excitations on the graph
        self.eLambda = np.zeros((self.V,self.S))
        
        self.margG = dict()
        for gene in self.genes:
            self.margG[gene] = [dict() for x in range(self.G)]
        
        self.elbo = 0.
    
    
    def update_lambdak(self,k):   
        ''' Parameter updates lambdak. '''
        self.alphak_s[k] = self.alpha0 + self.S
        self.betak_s[k] = self.beta0 + self.expGamma[k,:].sum()
    
    def update_exp_lambdak(self,g):
        ''' Update expectation lambdak. '''
        self.exp_lambdak[g] = self.alphak_s[g]/self.betak_s[g]
        self.exp_loglambdak[g] = digamma(self.alphak_s[g]) - math.log(self.betak_s[g])
    
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
    
        return (factorGraph, unitigFactorNodes, tempGraph)
    
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
    
    def parseFGString(self, factorGraph, outputString):
        mapVar = {}
        lines = outputString.split('\\n')
        bFirst = True
        for line in lines:
            if not bFirst:
                toks = line.split(',')
                if len(toks) == 2:
                    varIdx = int(toks[0][1:])
                    varName = factorGraph.varNames[int(varIdx)]
                    varValue = int(toks[1])
                    mapVar[varName] = varValue
            bFirst = False 
        return mapVar
    
    def updateUnitigFactors(self, unitigs, unitigMap, unitigFacNodes, gidx, tau):
        
        mapGammaG = self.expGamma[gidx,:]
        mapGammaG2 = self.expGamma2[gidx,:]
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
                
                if not self.BIAS:
                    T1 = self.M_train[v_idx,:]*mapGammaG*(lengthNode*currELambda - self.X[v_idx,:])
                else:
                    T1 = self.M_train[v_idx,:]*mapGammaG*(lengthNode*currELambda*self.expTheta2[v_idx] - self.X[v_idx,:]*self.expTheta[v_idx])
                
                dVal1 = 2.0*np.sum(T1)
                dVal2 = lengthNode*dSum2
                if self.BIAS:
                    dVal2 *= self.expTheta2[v_idx]
                    
                    
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

    def updateUnitigFactorsMarg(self,unitigs, unitigMap, unitigFacNodes, marg):

        for unitig in unitigs:
            if unitig in unitigFacNodes:
                
                unitigFacNode = unitigFacNodes[unitig]
                v_idx = unitigMap[unitig]
                P = unitigFacNode.P
                
                if unitig in marg:
                    unitigFacNode.P = np.copy(marg[unitig])
    
    def setPhiConstant(self,unitigs, unitigMap, unitigFacNodes):
        dLogNPhi = 0.
        
        for unitig in unitigs:
            if unitig in unitigFacNodes:
                
                unitigFacNode = unitigFacNodes[unitig]
                v_idx = unitigMap[unitig]

                P = unitigFacNode.P
                wmax = P.shape[0]
                self.logPhiPrior[v_idx] = math.log(1.0/wmax)        
    
    def updateUnitigFactorsRef(self, unitigs, unitigMap, unitigFacNodes, mapRef,ref):

        for unitig in unitigs:
            if unitig in unitigFacNodes:
                unitigFacNode = unitigFacNodes[unitig]
                v_idx = unitigMap[unitig]
                P = unitigFacNode.P
                tempMatrix = np.zeros_like(P)
                
                if ref in mapRef[unitig]:
                    delta = mapRef[unitig][ref]
                else:
                    delta = 10.0
                    
                tempMatrix[1] = 0.99*np.exp(-0.25*delta)
                tempMatrix[0] = 1.0 - tempMatrix[1]

                unitigFacNode.P = tempMatrix


    def removeGamma(self,g_idx):
        
        meanAss = self.expPhi[:,g_idx]
        gammaG  = self.expGamma[g_idx,:]
        
        self.eLambda -= meanAss[:,np.newaxis]*gammaG[np.newaxis,:]

    def addGamma(self,g_idx):
        
        meanAss = self.expPhi[:,g_idx]
        gammaG  = self.expGamma[g_idx,:]
        
        self.eLambda += meanAss[:,np.newaxis]*gammaG[np.newaxis,:]

    def updateTheta(self):
        
        self.eLambda = np.dot(self.expPhi, self.expGamma)
        
        self.tauTheta = self.expTau*self.exp_square_lambda()*self.lengths*self.lengths + self.tauTheta0 
        
        numer =  self.expTau*self.lengths*np.sum(self.X*self.eLambda,axis=1) + self.muTheta0*self.tauTheta0 
        
        self.muTheta = numer/self.tauTheta
    
        self.expTheta = np.asarray(TN_vector_expectation(self.muTheta,self.tauTheta))
        
        self.varTheta = np.asarray(TN_vector_variance(self.muTheta,self.tauTheta))

        self.expTheta2 = self.varTheta + self.expTheta*self.expTheta

    def updateGamma(self,g_idx):
        
        temp = np.delete(self.expGamma,g_idx,0)
        temp2 = np.delete(self.expPhi,g_idx,1)
       
        if not self.BIAS:       
            numer = self.M_train*(self.X - np.dot(temp2,temp)*self.lengths[:,np.newaxis])
        else:
            numer = self.M_train*(self.X*self.expTheta[:,np.newaxis] - np.dot(temp2,temp)*self.lengths[:,np.newaxis]*self.expTheta2[:,np.newaxis])   
        
        gphi = self.expPhi[:,g_idx]*self.lengths
        
        numer = gphi[:,np.newaxis]*numer

        denom = self.lengths*self.lengths*self.expPhi2[:,g_idx]
        if self.BIAS:
            denom *= self.expTheta2
                
        denom = denom[:,np.newaxis]*self.M_train
        dSum = np.sum(denom,0)
        nSum = np.sum(numer,0)
        
        lamb = 1.0/self.epsilon
        if self.ARD:
            lamb = self.exp_lambdak[g_idx] 
            
        nSum -= lamb/self.expTau

        muGammaG = nSum/dSum  
        tauGammaG = self.expTau*dSum

        expGammaG = np.asarray(TN_vector_expectation(muGammaG,tauGammaG))
        
        varGammaG = np.asarray(TN_vector_variance(muGammaG,tauGammaG))

        expGamma2G = varGammaG + expGammaG*expGammaG

        self.expGamma[g_idx,:]  = expGammaG
        self.expGamma2[g_idx,:] = expGamma2G
        self.tauGamma[g_idx,:]  = tauGammaG
        self.muGamma[g_idx,:]   = muGammaG
        self.varGamma[g_idx,:]  = varGammaG
        
    def updateTau(self):
        
        alphaD = self.alpha + 0.5*self.Omega
        betaD = self.beta + self.exp_square_diff()  
        
        self.alphaTau = alphaD
        self.betaTau = betaD
        
        self.expTau = alphaD/betaD
        self.expLogtau = digamma(alphaD) - math.log(betaD)

    def updateExpPhi(self,unitigs,mapUnitig,marg,g_idx):
    
        for unitig in unitigs:
        
            if unitig in marg:
                v_idx = mapUnitig[unitig]
                ND = marg[unitig].shape[0]
                self.expPhi[v_idx,g_idx] = np.sum(marg[unitig]*np.arange(ND))
                d2 = np.square(np.arange(ND))
                self.expPhi2[v_idx,g_idx] = np.sum(marg[unitig]*d2)
                
                self.HPhi[v_idx,g_idx] = ss.entropy(marg[unitig])
                
    def update(self, maxIter, removeRedundant,logFile=None):

        iter = 0
   
        self.expTau = 0.001 
        while iter < maxIter:
            #update phi marginals
            if removeRedundant:
                if iter > 50 and iter % 10 == 0:
                    self.removeRedundant(0.005, 10)
            
            for g in range(self.G):
                
                self.removeGamma(g)
                fgFileStubs = {}
                threads = []
                gidx = 0
                
                for gene, factorGraph in self.factorGraphs.items():
                    unitigs = self.assemblyGraphs[gene].unitigs
                   
                    self.updateUnitigFactors(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], g, self.expTau)
                    
                    factorGraph.reset()
        
                    factorGraph.var['zero+source+'].condition(1)

                    factorGraph.var['sink+infty+'].condition(1)
                    
                    graphString = str(factorGraph)
                    graphFileStub = str(uuid.uuid4()) + 'graph_'+ str(g) 
                    graphFileName = graphFileStub + '.fg'
                    
                    with open(graphFileName, "w") as text_file:
                        print(graphString, file=text_file)
                    
                    fgFileStubs[gene] = graphFileStub
                    
                    gidx += 1
                pool = ThreadPool(len(self.genes))
                results = []
                for gene, graphFileStub in fgFileStubs.items():
                    graphFileName = graphFileStub + '.fg'
                    outFileName = graphFileStub + '.out'
                    cmd = './runfg_marg ' + graphFileName + ' ' + outFileName + ' 0'
                    results.append(pool.apply_async(call_proc, (cmd,)))
                pool.close()
                pool.join()
                for result in results:
                    out, err = result.get()
                    print("out: {} err: {}".format(out, err))
                      
                for gene, factorGraph in self.factorGraphs.items():
                
                    outFile = fgFileStubs[gene] + '.out'
                
                    with open (outFile, "r") as infile:
                        outString = infile.readlines()
               
                    margP = self.parseMargString(factorGraph,str(outString))
                    if len(margP) > 0:
                        self.margG[gene][g] = self.parseMargString(factorGraph,str(outString)) 
       
                    self.updateExpPhi(unitigs,self.mapGeneIdx[gene],self.margG[gene][g],g)
       
                    os.remove(outFile)
                    os.remove(fgFileStubs[gene]  + '.fg')
                    
                self.addGamma(g)
            
            if self.ARD:
                for g in range(self.G):
                    self.update_lambdak(g)
                    self.update_exp_lambdak(g)
            
            
            for g in range(self.G):
                self.updateGamma(g)

            self.eLambda = np.zeros((self.V,self.S))
            for g in range(self.G):
                self.addGamma(g)
            
            self.updateTau()
            
            if self.BIAS:
                self.updateTheta()
            
            total_elbo = self.calc_elbo()    
            DivF = self.divF()
            Div  = self.div()
            print(str(iter)+ "," + str(Div) + "," + str(DivF)+ "," + str(total_elbo))

            if logFile is not None:
                with open(logFile, 'a') as logF:            
                    logF.write(str(iter)+","+ str(DivF)+ "," + str(total_elbo) + "\n")
            iter += 1
    
    def updateGammaFixed(self, maxIter, tau = None):
    
        iter = 0
   
        if tau is not None:
            self.expTau = tau
        else:
            self.expTau = 0.001
         
        while iter < maxIter:
            #update phi marginals
            
            for g in range(self.G):
                
                self.removeGamma(g)
            
                for gene, factorGraph in self.factorGraphs.items():
                    unitigs = self.assemblyGraphs[gene].unitigs
                   
                    
                    self.updateUnitigFactors(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], g, self.expTau)
                    
        
                    factorGraph.reset()
        
                    factorGraph.var['zero+source+'].condition(1)

                    factorGraph.var['sink+infty+'].condition(1)
                    
                    graphString = str(factorGraph)
                    graphFileName = str(uuid.uuid4()) + 'graph_'+ str(g) + '.fg'                    
                
                    with open(graphFileName, "w") as text_file:
                        print(graphString, file=text_file)
                
                    cmd = './runfg_marg ' + graphFileName + ' 0'
                
                    p = Popen(cmd, stdout=PIPE,shell=True)
        
                    outString = p.stdout.read()
               
                    margP = self.parseMargString(factorGraph,str(outString))
                    if len(margP) > 0:
                        self.margG[gene][g] = self.parseMargString(factorGraph,str(outString)) 
       
                    self.updateExpPhi(unitigs,self.mapGeneIdx[gene],self.margG[gene][g],g)
                    os.remove(graphFileName) 
                self.addGamma(g)
            
            if tau is not None:            
                self.updateTau()
            
            total_elbo = self.calc_elbo()    
            print(str(iter)+","+ str(self.divF()) +"," + str(total_elbo))  
            iter += 1
    
    
    def updatePhiFixed(self, maxIter):
    
        iter = 0
    
        while iter < maxIter:
            
            for g in range(self.G):
                self.updateGamma(g)

            self.eLambda = np.zeros((self.V,self.S))
            for g in range(self.G):
                self.addGamma(g)

            print(str(iter)+","+ str(self.divF()))  
            iter += 1
    
    def div(self):
        """Compute divergence of target matrix from its NMF estimate."""
        Va = self.eLambda
        return (np.multiply(self.XN*self.M_train, np.log(elop(self.XN, Va, truediv))) + self.M_train*(Va - self.XN)).sum()

    def divF(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""
        R = self.M_train*(self.eLambda - self.XN)
        return np.multiply(R, R).sum()/np.sum(self.M_train)

    def convertMAPToPath(self,mapPath,factorGraph):
    
        path = []
        visited = set()
        current = self.sourceNode
    
        while current != self.sinkNode:
            
            outPaths = list(factorGraph.successors(current))
            
            for outPath in outPaths:
                if mapPath[outPath] == 1 and outPath not in visited:
                    visited.add(outPath)
                    break
            
            path.append(current)
            
            current = list(factorGraph.successors(outPath))[0]

        return path


    def convertMAPMarg(self,MAP,mapNodes):
        marg = {}

        for unitig,assign in MAP.items():
            if unitig in mapNodes:
                unitigNode = mapNodes[unitig] 
                nN = unitigNode.dim
                
                tempMatrix = np.zeros(nN)
                tempMatrix[int(assign)] = 1.0
                marg[unitig] = tempMatrix
        return marg

    def initNMF(self):
        
        covNMF =  NMF(self.XN,self.M_train,self.G,n_run = 10, prng = self.prng)
    
        covNMF.factorize()
        covNMF.factorizeH()

        self.expGamma = np.copy(covNMF.H)
        self.expGamma2 = self.expGamma*self.expGamma
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
                stubName = str(uuid.uuid4()) + 'graph_'+ str(g)
                graphFileName = stubName + '.fg'
                outFileName = stubName + '.out'
                with open(graphFileName, "w") as text_file:
                    print(graphString, file=text_file)
                
                cmd = './runfg_marg ' + graphFileName + ' ' + outFileName + ' 0'
               
                #subprocess.run('./runfg_marg_old ' + graphFileName + ' 0 > ' + outFileName, shell=True,check=True)
 
                subprocess.run(cmd, shell=True,check=True)

                #p = Popen(cmd, stdout=PIPE,shell=True)
                
                with open (outFileName, "r") as myfile:
                    outString=myfile.readlines()
                
                self.margG[gene][g] = self.parseMargString(factorGraph,str(outString))
                self.updateExpPhi(unitigs,self.mapGeneIdx[gene],self.margG[gene][g],g)
                os.remove(graphFileName)
                os.remove(outFileName)
            self.addGamma(g)    
        print("-1,"+ str(self.div())) 

    def initNMFGamma(self,gamma):
        
        covNMF =  NMF(self.XN,self.M_train,self.G,n_run = 10, prng = self.prng)
        covNMF.random_initialize() 
        covNMF.H = np.copy(gamma)
        covNMF.factorizeW()
        self.eLambda = np.zeros((self.V,self.S))
        initEta = covNMF.W
            
        for g in range(self.G):
            
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.assemblyGraphs[gene].unitigs
                    
                self.updateUnitigFactorsW(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], initEta, g)
                  
                factorGraph.reset()
        
                factorGraph.var['zero+source+'].condition(1)

                factorGraph.var['sink+infty+'].condition(1)
                    
                graphString = str(factorGraph)
                graphFileName = str(uuid.uuid4()) + 'graph_'+ str(g) + '.fg'                
                outFileName = str(uuid.uuid4()) + 'graph_'+ str(g) + '.out'
                with open(graphFileName, "w") as text_file:
                    print(graphString, file=text_file)
                      
                cmd = './runfg_marg ' + graphFileName + ' ' + outFileName + ' 0'
                
                p = Popen(cmd, stdout=PIPE,shell=True)
                
                with open (outFileName, "r") as myfile:
                    outString=myfile.readlines()
                
                self.margG[gene][g] = self.parseMargString(factorGraph,str(outString))
             
                self.updateExpPhi(unitigs,self.mapGeneIdx[gene],self.margG[gene][g],g)
                os.remove(graphFileName) 
            self.addGamma(g)    
        print("-1,"+ str(self.div())) 


    def exp_square_lambda(self):
        ''' Compute: sum_s E_q(phi,gamma) [ sum ( Phi_v Gamma_s )^2 ]. '''
        
        eLambda2Sum = self.eLambda*self.eLambda
        
        diagonal = np.dot(self.expPhi*self.expPhi,self.expGamma*self.expGamma)
        
        return np.sum(eLambda2Sum - diagonal + np.dot(self.expPhi2,self.expGamma2), axis = 1)

    def exp_square_diff_matrix(self): 
        ''' Compute: sum_Omega E_q(phi,gamma) [ ( Xvs - L_v Phi_v Gamma_s )^2 ]. '''
        #return (self.M *( ( self.R - numpy.dot(self.exp_U,self.exp_V.T) )**2 + \
        #                  ( numpy.dot(self.var_U+self.exp_U**2, (self.var_V+self.exp_V**2).T) - numpy.dot(self.exp_U**2,(self.exp_V**2).T) ) ) ).sum()
        
        R = self.X - self.lengths[:,np.newaxis]*self.eLambda
        t1 = np.dot(self.expPhi*self.expPhi, self.expGamma*self.expGamma)
        diff = np.dot(self.expPhi2,self.expGamma2) - t1
        L2 = self.lengths*self.lengths
        diff2 = L2[:,np.newaxis]*diff
        
        return R*R + diff2

    def exp_square_diff(self): 
        ''' Compute: sum_Omega E_q(phi,gamma) [ ( Xvs - L_v Phi_v Gamma_s )^2 ]. '''
        #return (self.M *( ( self.R - numpy.dot(self.exp_U,self.exp_V.T) )**2 + \
        #                  ( numpy.dot(self.var_U+self.exp_U**2, (self.var_V+self.exp_V**2).T) - numpy.dot(self.exp_U**2,(self.exp_V**2).T) ) ) ).sum()
        
        R = self.X - self.lengths[:,np.newaxis]*self.eLambda
        t1 = np.dot(self.expPhi*self.expPhi, self.expGamma*self.expGamma)
        diff = np.dot(self.expPhi2,self.expGamma2) - t1
        L2 = self.lengths*self.lengths
        diff2 = L2[:,np.newaxis]*diff
        
        return np.sum(self.M_train*(R*R + diff2))

    def calc_elbo(self):
        ''' Compute the ELBO. '''
        total_elbo = 0.
        
        # Log likelihood               
        total_elbo += 0.5*self.Omega*(self.expLogtau - math.log(2*math.pi) ) 
        total_elbo -= 0.5*self.expTau*self.exp_square_diff()

        # Prior lambdak, if using ARD, and prior U, V
        if self.ARD:
            
            total_elbo += self.alpha0 * math.log(self.beta0) - sp.special.gammaln(self.alpha0) \
                          + (self.alpha0 - 1.)*self.exp_loglambdak.sum() - self.beta0 * self.exp_lambdak.sum()
            
            total_elbo += self.S * np.log(self.exp_lambdak).sum() - (self.exp_lambdak[:,np.newaxis] * self.expGamma).sum()
            
        else:
            total_elbo += np.sum(-math.log(self.epsilon) - self.expGamma/self.epsilon)
            
        #add phio prior assuming uniform 
        total_elbo += self.G*np.sum(self.logPhiPrior)
        
        #add tau prior
        total_elbo += self.alpha * math.log(self.beta) - sps.gammaln(self.alpha) 
        total_elbo += (self.alpha - 1.)*self.expLogtau - self.beta*self.expTau

        # q for lambdak, if using ARD
        if self.ARD:
            total_elbo += - sum([v1*math.log(v2) for v1,v2 in zip(self.alphak_s,self.betak_s)]) + sum([sp.special.gammaln(v) for v in self.alphak_s]) \
                          - ((self.alphak_s - 1.)*self.exp_loglambdak).sum() + (self.betak_s * self.exp_lambdak).sum()

        #add q for gamma
        qGamma = -0.5*np.log(self.tauGamma).sum() + 0.5*self.G*self.S*math.log(2.*math.pi)
        qGamma += np.log(0.5*sps.erfc(-self.muGamma*np.sqrt(self.tauGamma)/math.sqrt(2.))).sum()
        qGamma += (0.5*self.tauGamma * ( self.varGamma + (self.expGamma - self.muGamma)**2 ) ).sum()

        total_elbo += qGamma
        # q for tau
        total_elbo += - self.alphaTau * math.log(self.betaTau) + sps.gammaln(self.alphaTau) 
        total_elbo += - (self.alphaTau - 1.)*self.expLogtau + self.betaTau * self.expTau
        # q for phi
        total_elbo += np.sum(self.HPhi)
        return total_elbo

    def predict(self, M_pred):
        ''' Predict missing values in R. '''
        R_pred = self.lengths[:,np.newaxis]*np.dot(self.expPhi, self.expGamma)
        MSE = self.compute_MSE(M_pred, self.X, R_pred)
        #R2 = self.compute_R2(M_pred, self.R, R_pred)    
        #Rp = self.compute_Rp(M_pred, self.R, R_pred)        
        return MSE

    ''' Functions for computing MSE, R^2 (coefficient of determination), Rp (Pearson correlation) '''
    def compute_MSE(self,M,R,R_pred):
        ''' Return the MSE of predictions in R_pred, expected values in R, for the entries in M. '''
        return (M * (R-R_pred)**2).sum() / float(M.sum())


    def calcPathDist(self):
        
        dist = np.zeros((self.G,self.G))
        
        self.MAPs = defaultdict(list)
        
        pathsg = defaultdict(dict)
        
        for g in range(self.G):
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.assemblyGraphs[gene].unitigs
                self.updateUnitigFactorsMarg(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], self.margG[gene][g])

                factorGraph.reset()

                factorGraph.var['zero+source+'].condition(1)

                factorGraph.var['sink+infty+'].condition(1)

                graphString = str(factorGraph)
                graphFileName = str(uuid.uuid4()) + 'graph_'+ str(g) + '.fg'                

                with open(graphFileName, "w") as text_file:
                    print(graphString, file=text_file)

                cmd = './runfg ' + graphFileName + ' 0'

                p = Popen(cmd, stdout=PIPE,shell=True)

                outString = p.stdout.read()

                self.MAPs[gene].append(self.parseFGString(factorGraph,str(outString)))
                biGraph = self.factorDiGraphs[gene]

                pathG = self.convertMAPToPath(self.MAPs[gene][g],biGraph)
                pathG.pop(0)
                pathsg[g][gene] = pathG
                os.remove(graphFileName)
        for g in range(self.G):
            for h in range(g+1,self.G):
                diff = 0
                for gene in self.genes:
                    diff += len(set(pathsg[g][gene]) ^ set(pathsg[h][gene]))
                dist[g,h] = diff     

        return dist
        
    ''' Removes strain below a given total intensity and degenerate'''
    def removeRedundant(self, minIntensity, gammaIter):
    
        #calculate number of good strains
        nNewG = 0
        
        sumIntensity = np.max(self.expGamma,axis=1)
        
        dist = self.calcPathDist()
    #    dist = np.ones((self.G,self.G))
        removed = sumIntensity < minIntensity
        
        for g in range(self.G):
        
            if removed[g] == False:    

                for h in range(g+1,self.G):
                    if dist[g,h] == 0 and removed[h] == False:
                        removed[h] = True    
        
       
        retained = np.logical_not(removed) 
        nNewG = np.sum(retained)
        if nNewG < self.G:
            print("New strain number " + str(nNewG))
            self.G = nNewG
            newPhi  = self.expPhi[:,retained]        
            newPhi2 = self.expPhi2[:,retained]
            newPhiH = self.HPhi[:,retained] 

            newExpGamma = self.expGamma[retained,:]
            newExpGamma2 = self.expGamma2[retained,:]
            newMuGamma   = self.muGamma[retained,:]     
            newTauGamma  = self.tauGamma[retained,:]
            newVarGamma  = self.varGamma[retained,:]
        
            self.expPhi  = newPhi
            self.expPhi2 = newPhi2
            self.HPhi    = newPhiH
        
            self.expGamma  = newExpGamma
            self.expGamma2 = newExpGamma2
            self.muGamma   = newMuGamma
            self.tauGamma  = newTauGamma
            self.varGamma  = newVarGamma
        
            if self.ARD:
                self.alphak_s = self.alphak_s[retained]
                self.betak_s  = self.betak_s[retained]
                self.exp_lambdak = self.exp_lambdak[retained]
                self.exp_loglambdak = self.exp_loglambdak[retained]
        
            self.G       = nNewG
        
            iter = 0    
            while iter < gammaIter:
        
                if self.ARD:
                    for g in range(self.G):
                        self.update_lambdak(g)
                        self.update_exp_lambdak(g)
            
                for g in range(self.G):
                    self.updateGamma(g)

                self.eLambda = np.zeros((self.V,self.S))
                for g in range(self.G):
                    self.addGamma(g)
            
                self.updateTau()
            
                iter += 1
        
        self.margG = dict()
        for gene in self.genes:
            self.margG[gene] = [dict() for x in range(self.G)]       
 

    def collapseDegenerate(self):
        dist = np.zeros((self.G,self.G))
        self.MAPs = defaultdict(list)
        
        pathsg = defaultdict(dict)
        
        for g in range(self.G):
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.assemblyGraphs[gene].unitigs

                self.updateUnitigFactorsMarg(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], self.margG[gene][g])

                factorGraph.reset()

                factorGraph.var['zero+source+'].condition(1)

                factorGraph.var['sink+infty+'].condition(1)

                graphString = str(factorGraph)
                graphFileName = str(uuid.uuid4()) + 'graph_'+ str(g) + '.fg'                

                with open(graphFileName, "w") as text_file:
                    print(graphString, file=text_file)

                cmd = './runfg ' + graphFileName + ' 0'

                p = Popen(cmd, stdout=PIPE,shell=True)

                outString = p.stdout.read()

                self.MAPs[gene].append(self.parseFGString(factorGraph,str(outString)))
                biGraph = self.factorDiGraphs[gene]

                pathG = self.convertMAPToPath(self.MAPs[gene][g],biGraph)
                pathG.pop(0)
                pathsg[g][gene] = pathG
                os.remove(graphFileName)
        for g in range(self.G):
            for h in range(g+1,self.G):
                diff = 0
                for gene in self.genes:
                    diff += len(set(pathsg[g][gene]) ^ set(pathsg[h][gene]))
                dist[g,h] = diff 
        
        
        collapsed = [False]*self.G 
        for g in range(self.G):
            
            for h in range(g+1,self.G):
                if dist[g,h] == 0 and collapsed[h] == False:
                    self.expGamma[g,:] += self.expGamma[h,:]
                    self.expGamma[h,:] = 0.
                    self.expGamma2[h,:] = 0.
                    self.muGamma[h,:] = 0.
                    collapsed[h] = True
        
        

    def average_MSE_CV(self):

        dSumE = 0
        for n in range(self.no_folds):
            self.M_train = self.M_trains[n]
            self.M_test = self.M_tests[n]
        
            self.initNMF()
            
            self.update(50)
            
            dErr = self.predict(self.M_test)

            dSumE += dErr
        dMeanE = dSumE/float(self.no_folds)
        print(str(self.G) + ",MeanE=" + str(dMeanE))

    def writeMarginals(self,fileName):

        with open(fileName, "w") as margFile:
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.assemblyGraphs[gene].unitigs

                for unitig in unitigs:
                    if unitig in self.margG[gene][0]:
                        vals = []
                        for g in range(self.G):
                            vals.append(str(self.margG[gene][g][unitig][1]))
                        vString = ",".join(vals)
                    
                        margFile.write(gene + "_" + unitig + "," + vString + "\n")

    def writeTheta(self,fileName):

        with open(fileName, "w") as thetaFile:
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.assemblyGraphs[gene].unitigs

                for unitig in unitigs:
                    if unitig in self.mapGeneIdx[gene]:
                        
                        v = self.mapGeneIdx[gene][unitig]
                        
                        thetaFile.write(gene + "_" + unitig + "," + str(self.expTheta[v]) + "\n")


    def writeMaximals(self,fileName):

        with open(fileName, "w") as margFile:
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.assemblyGraphs[gene].unitigs

                for unitig in unitigs:
                    if unitig in self.margG[gene][0]:
                        vals = []
                        for g in range(self.G):
                            if self.MAPs[gene][g][unitig] == 1:
                                vals.append(g)                  
                        vString = "\t".join([str(x) for x in vals])

                        margFile.write(gene + "_" + unitig + "\t" + vString + "\n")


    def getMaximalUnitigs(self,fileName):

        self.MAPs = defaultdict(list)
        haplotypes = defaultdict(list)

        output = np.sum(self.expGamma,axis=1) > 0.01

        for g in range(self.G):
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.assemblyGraphs[gene].unitigs

                self.updateUnitigFactorsMarg(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], self.margG[gene][g])        
                factorGraph.reset()

                factorGraph.var['zero+source+'].condition(1)

                factorGraph.var['sink+infty+'].condition(1)

                graphString = str(factorGraph)
                graphFileName = str(uuid.uuid4()) + 'graph_'+ str(g) + '.fg'                

                with open(graphFileName, "w") as text_file:
                    print(graphString, file=text_file)

                cmd = './runfg ' + graphFileName + ' 0'

                p = Popen(cmd, stdout=PIPE,shell=True)

                outString = p.stdout.read()

                self.MAPs[gene].append(self.parseFGString(factorGraph,str(outString)))
                biGraph = self.factorDiGraphs[gene]
                
                pathG = self.convertMAPToPath(self.MAPs[gene][g],biGraph)
                pathG.pop(0)
                unitig = self.assemblyGraphs[gene].getUnitigWalk(pathG)
                haplotypes[gene].append(unitig)
                os.remove(graphFileName)
        with open(fileName, "w") as fastaFile:
            for g in range(self.G):
                if output[g]:
                    for gene, factorGraph in self.factorGraphs.items():
                        fastaFile.write(">" + str(gene) + "_" + str(g) + "\n")
                        fastaFile.write(haplotypes[gene][g]+"\n")

    def outputOptimalRefPaths(self, ref_hit_file):
        
        (refHits,allHits) = readRefAssign(ref_hit_file)
        NR = len(allHits)
        refMAPs = [None]*NR
        r = 0
        refs = []
        self.margG = {}
        for ref in allHits:
            
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.assemblyGraphs[gene].unitigs
                
                self.updateUnitigFactorsRef(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], refHits, ref)
            
                factorGraph.reset()
        
                factorGraph.var['zero+source+'].condition(1)

                factorGraph.var['sink+infty+'].condition(1)

                graphString = str(factorGraph)
                graphFileName = str(uuid.uuid4()) + 'graph_'+ str(g) + '.fg'                

                with open(graphFileName, "w") as text_file:
                    print(graphString, file=text_file)

                cmd = './runfg ' + graphFileName + ' 0'

                p = Popen(cmd, stdout=PIPE,shell=True)

                outString = p.stdout.read()

                refMAPs[r] = self.parseFGString(factorGraph, str(outString))
            
                self.margG[gene][ref] = self.convertMAPMarg(refMAPs[r],factorGraph.mapNodes)
                self.updateExpPhi(unitigs,self.mapGeneIdx[gene],self.margG[gene][ref],r)
        
                biGraph = self.factorDiGraphs[gene]
                pathG = self.convertMAPToPath(refMAPs[r], biGraph)
                pathG.pop(0)
                unitig = self.assemblyGraphs[gene].getUnitigWalk(pathG)
                    
                refs.append(unitig)    
                os.remove(graphFileName)
            r = r + 1
        r = 0
        with open("Ref.fa", "w") as fastaFile:
            for ref in allHits:
                fastaFile.write(">" + ref + "\n")
                fastaFile.write(refs[r]+"\n")
                r = r + 1

    def writeGammaMatrix(self, gammaFile):

        with open(gammaFile, "w") as gammaFile:
            for g in range(self.G):
                gammaVals = self.expGamma[g,:].tolist()
                
                gString = ",".join([str(x) for x in gammaVals])

                gammaFile.write(str(g) + "," + gString + "\n")


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("unitig_file", help="unitig fasta file in Bcalm2 format")

    parser.add_argument("cov_file", help="coverage file")

    parser.add_argument("kmer_length", help="kmer length assumed overlap")

    parser.add_argument("outFileStub", help="stub for output file names")

    parser.add_argument('-fg','--gamma_file', nargs='?',help="gamma file")

    parser.add_argument('-fr','--ref_blast_file', nargs='?',help="ref blast file")

    parser.add_argument('-r','--read_length',nargs='?', default=100, type=int, 
        help=("read length"))

    parser.add_argument('-g','--strain_number',nargs='?', default=5, type=int, 
        help=("maximum number of strains"))

    parser.add_argument('-f','--frac',nargs='?', default=0.75, type=float, 
        help=("fraction for path source sink"))

    args = parser.parse_args()

#    import ipdb; ipdb.set_trace()

    np.random.seed(2)
    prng = RandomState(238329)
    
    if args.unitig_file.endswith('.gfa'):
        unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.unitig_file,int(args.kmer_length), args.cov_file)
    else:
        unitigGraph = UnitigGraph.loadGraph(args.unitig_file,int(args.kmer_length), args.cov_file)   
  
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
            
            (source_list, sink_list) = unitigSubGraph.selectSourceSinks(args.frac)
            
            #(source_list, sink_list) = unitigSubGraph.selectAllSourceSinks()

    #        source_list = [('465248',True),('465730',True),('466718',True)]
     #       sink_list = [('462250',True),('467494',True)]

            source_names = [convertNodeToName(source) for source in source_list] 
            sink_names = [convertNodeToName(sink) for sink in sink_list]
            
            sink_maps[str(c)] = sink_list
            source_maps[str(c)] = source_list
        c = c + 1

    assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G = args.strain_number, readLength=args.read_length, ARD=True)
    
    if args.ref_blast_file:
        refPath = assGraph.outputOptimalRefPaths(args.ref_blast_file)
                
        assGraph.updatePhiFixed(100)
        glm = GLM(distr='poisson')
        scaler = StandardScaler().fit(assGraph.expPhi)
        glm.fit(scaler.transform(assGraph.phiMean), assGraph.XN[:,0])

        # predict using fitted model on the test data
        yhat_test = glm.predict(scaler.transform(assGraph.phiMean))

        # score the model
        deviance = glm.score(assGraph.phiMean, assGraph.XN[:,0])

    if args.gamma_file:
        covs    = p.read_csv(args.gamma_file, header=0, index_col=0)
    
        covs.drop('Strain', axis=1, inplace=True)
    
        cov_matrix = covs.values
    
        assGraph.expGamma = cov_matrix/assGraph.readLength
        assGraph.expGamma2 = np.square(assGraph.muGamma)
        assGraph.initNMFGamma(assGraph.expGamma)
        #assGraph.tau = 1.0e-4
        assGraph.update(100)
    else:
        assGraph.initNMF()

        assGraph.update(100, True)
        
        assGraph.writeMarginals(args.outFileStub + "margFile.csv")
   
        assGraph.getMaximalUnitigs(args.outFileStub + "Haplo_" + str(assGraph.G) + ".fa")
 
        assGraph.writeMaximals(args.outFileStub + "maxFile.tsv")
   
        assGraph.writeGammaMatrix(args.outFileStub + "Gamma.csv") 

        errMatrix = assGraph.exp_square_diff_matrix()

        for v in range(assGraph.V):
            for s in range(assGraph.S):
                if errMatrix[v,s] > 0:
                    test = 1.0/errMatrix[v,s]
                    print(str(v) + "," + str(s) + "," + str(assGraph.X[v,s]) + "," + str(test))

 #       print("Debug")
if __name__ == "__main__":
    main(sys.argv[1:])


