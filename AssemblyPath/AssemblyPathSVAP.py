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
from scipy.stats import truncnorm
from scipy.special import erfc
from scipy.special import erf

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

from AssemblyPath.graph import Graph
from Utils.UtilsFunctions import convertNodeToName
from Utils.UtilsFunctions import elop
from Utils.UtilsFunctions import expNormLogProb
from Utils.UtilsFunctions import TN_vector_expectation
from Utils.UtilsFunctions import TN_vector_variance
from Utils.UtilsFunctions import readRefAssign
from Utils.UnitigGraph import UnitigGraph
from AssemblyPath.NMFM import NMF
 
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
    """ Class for structured Poisson variational approximation on Assembly Graph"""    
    minW = 1.0e-3    
    def __init__(self, prng, assemblyGraphs, source_maps, sink_maps, G = 2, maxFlux=2, 
                readLength = 100, epsilon = 1.0e5, alpha0=1.0e-9,beta0=1.0e-9, 
                ARD = False, BIAS = True, alphaTheta0 = 10.0, betaTheta0 = 10.0, alphaDelta0 = 1.0e-2, betaDelta0 = 1.0,
                minIntensity = None, fgExePath="./runfg_source/", working_dir="/tmp", minSumCov = 0.):
        self.prng = prng #random state to store

        self.readLength = readLength #sequencing read length
 
        self.assemblyGraphs = assemblyGraphs
        
        self.source_maps = source_maps
        
        self.sink_maps = sink_maps
 
        self.fgExePath = fgExePath

        self.factorGraphs = {} # dict of factorGraphs as pyfac Graphs

        self.factorDiGraphs = {} # dict of factorGraphs as networkx diGraphs

        self.unitigFactorNodes = {}
        
        self.unitigFluxNodes = {}
        
        self.maxFlux = 2
        
        self.working_dir = working_dir
        
        self.logTau = -100.0 # dummy intensity associated with phi = 0

        #define dummy source and sink node names
        self.sinkNode = 'sink+'
        
        self.sourceNode = 'source+'
        
        if minIntensity == None:
            self.minIntensity = 2.0/self.readLength
        else:
            self.minIntensity = minIntensity

        self.minSumCov = minSumCov

        #prior parameters for Gamma delta
        self.alphaDelta0 = alphaDelta0
        self.betaDelta0  = betaDelta0
        
        self.ARD = ARD
        if self.ARD:
            self.alpha0 = alpha0
            self.beta0 = beta0
            
        self.BIAS = BIAS
        if self.BIAS:
            self.alphaTheta0 = alphaTheta0
            self.betaTheta0 = betaTheta0

        self.V = 0
        self.mapIdx = {}
        self.mapUnitigs = {}
        self.adjLengths = {}
        self.covMapAdj = {}
        
        self.unitigs = []
        self.genes = []
        self.mapGeneIdx = collections.defaultdict(dict)
        bFirst = True
        for gene in sorted(assemblyGraphs):
            self.genes.append(gene)
            assemblyGraph = assemblyGraphs[gene]
            (factorGraph, unitigFactorNode, unitigFluxNode, factorDiGraph) = self.createFactorGraph(assemblyGraph, source_maps[gene], sink_maps[gene])
           
            unitigsDash = list(unitigFactorNode.keys())
            unitigsDash.sort(key=int) 
            self.factorGraphs[gene] = factorGraph

            self.factorDiGraphs[gene] = factorDiGraph 

            self.unitigFactorNodes[gene] = unitigFactorNode
            
            self.unitigFluxNodes[gene] = unitigFluxNode

            self.mapUnitigs[gene] = unitigsDash
            unitigAdj = [gene + "_" + s for s in unitigsDash]
            self.unitigs.extend(unitigAdj)
            for (unitigNew, unitig) in zip(unitigAdj,unitigsDash):
                if unitig.startswith('connect'):
                    self.adjLengths[unitigNew] = 0.0
                else:
                    #self.adjLengths[unitigNew] = assemblyGraph.lengths[unitig] - 2.0*assemblyGraph.overlapLength + 2.0*self.readLength
                    self.adjLengths[unitigNew] = assemblyGraph.lengths[unitig] - assemblyGraph.overlapLength  + self.readLength
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
        
        for gene, unitigFluxNode in self.unitigFluxNodes.items():
            self.removeNoise(unitigFluxNode, self.mapUnitigs[gene], gene, self.minSumCov)
        
        #create mask matrices
        self.Identity = np.ones((self.V,self.S))
        #Now initialise SVA parameters
        self.G = G
        self.Omega = self.V*self.S      
 
        self.expDelta = self.alphaDelta0/self.betaDelta0
        self.expLogDelta = digamma(self.alphaDelta0) - math.log(self.betaDelta0)
        self.aDelta = self.alphaDelta0
        self.bDelta = self.betaDelta0
        #list of mean assignments of strains to graph
        self.expPhi = np.zeros((self.V,self.G))
        self.expPhi2 = np.zeros((self.V,self.G))
        self.expLogPhi = np.zeros((self.V,self.G))
        self.HPhi = np.zeros((self.V,self.G))

        self.epsilon = epsilon #parameter for gamma exponential prior
        self.expGamma = np.zeros((self.G,self.S)) #expectation of gamma
        self.expGamma2 = np.zeros((self.G,self.S))
        
        self.aGamma = np.zeros((self.G,self.S))
        self.bGamma = np.zeros((self.G,self.S))
        
        self.expLogGamma = np.zeros((self.G,self.S))
        #current excitations on the graph not including bias
        self.eLambda = np.zeros((self.V,self.S))
        self.norm_p = np.zeros((self.V, self.S,self.G + 1))
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
            self.expTheta.fill(self.alphaTheta0/self.betaTheta0)
            
             
            self.aTheta = np.ones(self.V)
            self.aTheta.fill(self.alphaTheta0)
            
            self.bTheta = np.ones(self.V)
            self.bTheta.fill(self.betaTheta0)

            self.expLogTheta = digamma(self.alphaTheta0) - math.log(self.betaTheta0)
            self.varTheta = self.alphaTheta0/(self.betaTheta0*self.betaTheta0)
            
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
    
        (factorGraph, unitigFactorNodes, unitigFluxNodes) = self.generateFactorGraph(tempGraph, assemblyGraph.unitigs)
    
        return (factorGraph, unitigFactorNodes, unitigFluxNodes, tempGraph)
    
    def generateFactorGraph(self, factorGraph, unitigs):
        probGraph = Graph()

        
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
        unitigFluxNodes = {}
                
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
                
                unitigFluxNodes[unitig] = probGraph.addFacNode(fluxMatrix, *([probGraph.mapNodes[unitig]] + mapNodesF))
            
                discreteMatrix = np.zeros((nMax,1))
            
                unitigFacNodes[unitig] = probGraph.addFacNode(discreteMatrix, probGraph.mapNodes[unitig])
        
        return (probGraph,unitigFacNodes, unitigFluxNodes)
        
    def removeNoise(self, unitigFluxNodes, unitigs, gene, minSumCov):
        
        for unitig in unitigs:
            v_idx = self.mapGeneIdx[gene][unitig]
            sumCov = np.sum(self.X[v_idx,:])
            
            if sumCov < minSumCov:
                unitigFluxNodes[unitig].P = np.zeros_like(unitigFluxNodes[unitig].P)
                zeroth = next(np.ndindex(unitigFluxNodes[unitig].P.shape))
                unitigFluxNodes[unitig].P[zeroth] = 1.   
 
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
   
    def parseMargString(self, factorGraph, lines):
        mapMarg = {}
        lines = [x.strip() for x in lines] 
         
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
    
    def parseFGString(self, factorGraph, lines):
        mapVar = {}
        lines = [x.strip() for x in lines] 
        #lines = outputString.split('\\n')
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
    
    def updateUnitigFactors(self, unitigs, unitigMap, unitigFacNodes, gidx):
        
        mapGammaG = self.expGamma[gidx,:]
        mapLogGammaG = self.expLogGamma[gidx,:]
        #dSum2 = np.sum(mapGammaG2)
        tempLogGamma = np.zeros((self.G + 1, self.S))
        
        tempLogGamma[0,:]  = self.expLogDelta
        tempLogGamma[1:,:] = self.expLogGamma
        
        for unitig in unitigs:
            if unitig in unitigFacNodes:
                unitigFacNode = unitigFacNodes[unitig]
                v_idx = unitigMap[unitig]
                P = unitigFacNode.P
                nMax = unitigFacNode.P.shape[0]
                tempMatrix = np.zeros_like(P)
            
                #first comute fractions
                
                temp_p = np.zeros((self.S,self.G + 1)) #first component noise
                norm_p = np.zeros((self.S,self.G + 1))
                sum_p = np.zeros((self.S,self.G + 1))
                temp_p[:,0] = self.expLogDelta
                
                temp_v = self.expLogPhi[v_idx,:]
                temp_v[gidx] = 0.
                
                temp_p[:,1:(self.G + 1)] = np.transpose(self.expLogGamma) + temp_v[np.newaxis,:]
                
                temp_log_phi = np.ones(self.G + 1)
                temp_log_phi[1:] = np.copy(self.expLogPhi[v_idx,:])
                
                for d in range(nMax):
                    if d == 0:
                        temp_p[:,gidx + 1] = self.logTau 
                        temp_log_phi[gidx + 1] = self.logTau 
                    else:
                        temp_p[:,gidx + 1] = math.log(d)
                        temp_log_phi[gidx + 1] = math.log(d)
                    
                    for s in range(self.S):
                        norm_p[s,:] = expNormLogProb(temp_p[s,:])
                    
                    temp_s = np.sum(norm_p*(tempLogGamma.transpose() + temp_log_phi[np.newaxis,:]),axis=1)
                    
                    tempMatrix[d] = np.sum(-d*self.expTheta[v_idx]*mapGammaG*self.lengths[v_idx] + self.X[v_idx,:]*(temp_s))

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
        
        self.aTheta = self.alphaTheta0 + np.sum(self.X,axis=1)
        self.bTheta = self.alphaTheta0 + np.sum(self.eLambda,axis=1)
        
        self.expTheta = self.aTheta/self.bTheta

        self.varTheta = self.aTheta/(self.bTheta*self.bTheta)
    
    def updateP(self):
    
        temp_p = np.zeros((self.V, self.S,self.G + 1)) #first component noise
        
        temp_p1 = np.zeros((self.S,self.G + 1))
        
        temp_p1[:,0] = self.expLogDelta
        
        temp_p1[:,1:self.G + 1] = np.transpose(self.expLogGamma)
        temp_v = np.zeros((self.V,self.G + 1))
        temp_v[:,1:self.G + 1] = self.expLogPhi
        
        temp_p = temp_p1[np.newaxis,:,:] + temp_v[:,np.newaxis,:]
        
        for v in range(self.V):
            for s in range(self.S):
                self.norm_p[v,s,:] = expNormLogProb(temp_p[v,s,:])
        

    def updateGammaDelta(self):
        
        self.updateP()
           
        bTemp = np.zeros(self.G + 1)

        aTempT = np.einsum('vsg,vs->sg',self.norm_p, self.X)
        aTemp = aTempT.transpose()
        
        tTheta = self.expTheta*self.lengths
        bTemp[0] = np.sum(tTheta)

        bTemp[1:self.G + 1] = np.sum(tTheta[:,np.newaxis]*self.expPhi,axis=0)

        lamb = 1.0/self.epsilon
        if self.ARD:
            lamb = self.exp_lambdak 

        aTemp[1:self.G + 1,:] += 1.
        bTemp[1:self.G + 1] += lamb

        self.aGamma = aTemp[1:self.G + 1,:]
        self.bGamma = bTemp[1:self.G + 1]

        self.expGamma  = self.aGamma/self.bGamma[:,np.newaxis]
        b2Gamma = self.bGamma*self.bGamma
        self.varGamma  = self.aGamma/(b2Gamma[:,np.newaxis])
        
        self.expLogGamma = digamma(self.aGamma) - np.log(self.bGamma)[:,np.newaxis]
        
        self.aDelta = np.sum(aTemp[0,:]
        self.bDelta = bTemp[0]
        
        self.aDelta += self.alphaDelta0
        self.bDelta += self.betaDelta0 
        
        self.expDelta = self.aDelta/self.bDelta
        self.expLogDelta = digamma(self.aDelta) - np.log(self.bDelta)
        
        


    def updateExpPhi(self,unitigs,mapUnitig,marg,g_idx):
    
        for unitig in unitigs:
        
            if unitig in marg:
                v_idx = mapUnitig[unitig]
                ND = marg[unitig].shape[0]
                self.expPhi[v_idx,g_idx] = np.sum(marg[unitig]*np.arange(ND))
                d2 = np.square(np.arange(ND))
                self.expPhi2[v_idx,g_idx] = np.sum(marg[unitig]*d2)
                
                pseudoD = np.arange(ND,dtype=np.float)
                pseudoD[1:] = np.log(pseudoD[1:])
                pseudoD[0] = self.logTau
                self.expLogPhi[v_idx,g_idx] = np.sum(marg[unitig]*pseudoD)
                
                self.HPhi[v_idx,g_idx] = ss.entropy(marg[unitig])
                
    def writeFactorGraphs(self, g, drop_strain, relax_path):
        fgFileStubs = {}
        for gene, factorGraph in self.factorGraphs.items():
            
            if not drop_strain[gene][g]: 
            
                unitigs = self.assemblyGraphs[gene].unitigs
                   
                self.updateUnitigFactors(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], g)
                    
                factorGraph.reset()
        
                if relax_path:
                    factorGraph.var['zero+source+'].clear_condition()

                    factorGraph.var['sink+infty+'].clear_condition()
                else:
                    factorGraph.var['zero+source+'].condition(1)

                    factorGraph.var['sink+infty+'].condition(1)
                    
                graphString = str(factorGraph)
                graphFileStub = str(uuid.uuid4()) + 'graph_'+ str(g) 
                graphFileName = self.working_dir + "/" + graphFileStub + '.fg'
                    
                with open(graphFileName, "w") as text_file:
                    print(graphString, file=text_file)
                    
                fgFileStubs[gene] = graphFileStub
        
        return fgFileStubs
                    
    def readMarginals(self, fgFileStubs, g, drop_strain):
    
        for gene, factorGraph in self.factorGraphs.items():
            if not drop_strain[gene][g]: 
                outFile = self.working_dir + "/" + fgFileStubs[gene] + '.out'
        
                try:
                    inHandle = open(outFile, 'r')
                    
                    outLines = inHandle.readlines()

                    inHandle.close()

                    margP = self.parseMargString(factorGraph, outLines)
                    if len(margP) > 0:
                        self.margG[gene][g] = margP

                    self.updateExpPhi(self.assemblyGraphs[gene].unitigs,self.mapGeneIdx[gene],self.margG[gene][g],g)
        
                    if os.path.exists(outFile):
                        os.remove(outFile)

                    fgFile = self.working_dir + "/" + fgFileStubs[gene]  + '.fg'
                    if os.path.exists(fgFile):
                        os.remove(fgFile)
                except FileNotFoundError:
                
                    if nx.is_directed_acyclic_graph(self.factorDiGraphs[gene]):
                        print("Attempt greedy path")
                        greedyPath = self.sampleGreedyPath(gene, g)
                    
                        for unitig in self.assemblyGraphs[gene].unitigs:
                            v_idx = self.mapGeneIdx[gene][unitig]
                        
                            self.expPhi[v_idx,g] = 0.
                            self.expPhi2[v_idx,g] = 0.
                            self.expLogPhi[v_idx,g] = self.logTau
                        
                        for unitig in greedyPath:
                            if unitig in self.mapGeneIdx[gene]:
                                v_idx = self.mapGeneIdx[gene][unitig]
                        
                                self.expPhi[v_idx,g] = 1.
                                self.expPhi2[v_idx,g] = 1.
                                elf.expLogPhi[v_idx,g] = 0.
                    else:
                        print("Cannot attempt greedy path")
                        
                    fgFile = self.working_dir + "/" + fgFileStubs[gene]  + '.fg'
                    if os.path.exists(fgFile):
                        os.remove(fgFile)

    def update(self, maxIter, removeRedundant,logFile=None,drop_strain=None,relax_path=False):

        if drop_strain is None:
            drop_strain = {gene:[False]*self.G for gene in self.genes}
            
        iter = 0
        self.eLambda = np.dot(self.expPhi, self.expGamma)
        
        while iter < maxIter:
            #update phi marginals
            if removeRedundant:
                if iter > 50 and iter % 10 == 0:
                    self.removeRedundant(self.minIntensity, 10,relax_path)
            
            for g in range(self.G):
                
                self.removeGamma(g)
                fgFileStubs = {}
                threads = []
                
                fgFileStubs = self.writeFactorGraphs(g, drop_strain, relax_path)
                
                pool = ThreadPool(len(self.genes))
                results = []
                for gene, graphFileStub in fgFileStubs.items():
                    graphFileName = self.working_dir + '/' + graphFileStub + '.fg'
                    outFileName = self.working_dir + '/' + graphFileStub + '.out'
                    cmd = self.fgExePath + 'runfg_flex ' + graphFileName + ' ' + outFileName + ' 0 -1'
                    results.append(pool.apply_async(call_proc, (cmd,)))
                pool.close()
                pool.join()
                for result in results:
                    out, err = result.get()
        #            print("out: {} err: {}".format(out, err))
                
                self.readMarginals(fgFileStubs, g, drop_strain)
                           
                self.addGamma(g)
            
            if self.ARD:
                for g in range(self.G):
                    self.update_lambdak(g)
                    self.update_exp_lambdak(g)
            
            self.updateGammaDelta()

            self.eLambda = np.zeros((self.V,self.S))
            for g in range(self.G):
                self.addGamma(g)
            
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
    
    
    
    
    def updateGammaFixed(self, maxIter):
    
        iter = 0
         
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
                    graphFileName = self.working_dir + '/' + str(uuid.uuid4()) + 'graph_'+ str(g) + '.fg'                    
                
                    with open(graphFileName, "w") as text_file:
                        print(graphString, file=text_file)
                
                    cmd = './runfg_marg ' + graphFileName + ' 0'
                
                    p = Popen(cmd, stdout=PIPE,shell=True)
        
                    outLines = p.stdout.read()
               
                    margP = self.parseMargString(factorGraph, outLines)
                    if len(margP) > 0:
                        self.margG[gene][g] = margP
       
                    self.updateExpPhi(unitigs,self.mapGeneIdx[gene],self.margG[gene][g],g)
                    os.remove(graphFileName) 
                self.addGamma(g)
                      
            
            total_elbo = self.calc_elbo()    
            print(str(iter)+","+ str(self.divF()) +"," + str(total_elbo))  
            iter += 1
    
    
    def updatePhiFixed(self, maxIter):
    
        iter = 0
    
        while iter < maxIter:
            
            self.updateGamma()

            self.eLambda = np.zeros((self.V,self.S))
            for g in range(self.G):
                self.addGamma(g)

            print(str(iter)+","+ str(self.divF()))  
            iter += 1
    
    def div(self):
        """Compute divergence of target matrix from its NMF estimate."""
        Va = self.eLambda
        if self.BIAS:
            Va = self.expTheta[:,np.newaxis]*Va
            
        return (np.multiply(self.XN, np.log(elop(self.XN, Va, truediv))) + (Va - self.XN)).sum()

    def divF(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""
        
        if self.BIAS:
            R = self.expTheta[:,np.newaxis]*self.eLambda - self.XN
        else:
            R = self.eLambda - self.XN
            
        return np.multiply(R, R).sum()/self.Omega

    def divF_matrix(self):
        """Compute squared Frobenius norm of a target matrix and its NMF estimate."""

        if self.BIAS:
            R = self.expTheta[:,np.newaxis]*self.eLambda - self.XN
        else:
            R = self.eLambda - self.XN

        return np.multiply(R, R)

    def divergenceN(self, XN, Va):
    
        return (np.multiply(XN, np.log(elop(XN, Va, truediv))) - XN + Va).sum()


    def sampleGreedyPath(self, gene, g):
    
        path = []
    
        current = self.sourceNode
    
        biGraph = self.factorDiGraphs[gene]
    
        assert nx.is_directed_acyclic_graph(biGraph)
        
        while current != self.sinkNode:
            
            outPaths = list(biGraph.successors(current))
            
            destinations = [list(biGraph.successors(x))[0] for x in outPaths]
            path.append(current)
            NDest = len(destinations)
            if NDest > 0:
                n = 0
                divN = np.zeros(NDest)
                for dest in destinations:
                    unitig = dest[:-1]
                
                    if unitig in self.unitigFactorNodes[gene]:
                        unitigFacNode = self.unitigFactorNodes[gene][unitig]
                        divN[n] = unitigFacNode.P[1]

                    n = n+1 
                outPath = outPaths[np.argmax(divN)]
            else:
                break
            
            current = list(biGraph.successors(outPath))[0]
            #print(str(current))
        return path


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
            if mapPath[outPath] == 1: 
                current = list(factorGraph.successors(outPath))[0]
            else:
                break
            
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
        
        covNMF =  NMF(self.XN,self.Identity,self.G,n_run = 20, prng = self.prng)
    
        covNMF.factorize()
        covNMF.factorizeH()

        self.expGamma = np.copy(covNMF.H)
        self.expGamma2 = self.expGamma*self.expGamma
        self.expLogGamma = np.log(self.expGamma)
        covNMF.factorizeW()
        
        initEta = covNMF.W
        
        for g in range(self.G):
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.assemblyGraphs[gene].unitigs
                    
                self.updateUnitigFactorsW(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], initEta, g)
                  
                factorGraph.reset()
                print(gene) 
                factorGraph.var['zero+source+'].condition(1)

                factorGraph.var['sink+infty+'].condition(1)
                    
                graphString = str(factorGraph)
                stubName = str(uuid.uuid4()) + 'graph_'+ str(g)
                graphFileName = self.working_dir + '/' + stubName + '.fg'
                outFileName = self.working_dir + '/' + stubName + '.out'
                with open(graphFileName, "w") as text_file:
                    print(graphString, file=text_file)
                
                cmd = self.fgExePath + 'runfg_flex ' + graphFileName + ' ' + outFileName + ' 0 -1'
               
                #subprocess.run('./runfg_marg_old ' + graphFileName + ' 0 > ' + outFileName, shell=True,check=True)
 
                subprocess.run(cmd, shell=True,check=True)

                #p = Popen(cmd, stdout=PIPE,shell=True)
                
                with open (outFileName, "r") as myfile:
                    outLines=myfile.readlines()
                
                self.margG[gene][g] = self.parseMargString(factorGraph,outLines)
                self.updateExpPhi(unitigs,self.mapGeneIdx[gene],self.margG[gene][g],g)
                os.remove(graphFileName)
                os.remove(outFileName)
            self.addGamma(g)   
 
        self.updateGammaDelta()

        print("-1,"+ str(self.div())) 

    def initNMFGamma(self,gamma):
        
        covNMF =  NMF(self.XN,self.Identity,self.G,n_run = 10, prng = self.prng)
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
                outFileStub = str(uuid.uuid4()) + 'graph_'+ str(g)
                graphFileName = self.working_dir + '/' + outFileStub + '.fg'                
                outFileName = self.working_dir + '/' + outFileStub + '.out'
                with open(graphFileName, "w") as text_file:
                    print(graphString, file=text_file)
                      
                cmd = self.fgExePath + 'runfg_flex ' + graphFileName + ' ' + outFileName + ' 0 -1'                
                p = Popen(cmd, stdout=PIPE,shell=True)
                
                with open (outFileName, "r") as myfile:
                    outLines=myfile.readlines()
                
                self.margG[gene][g] = self.parseMargString(factorGraph, outLines)
             
                self.updateExpPhi(unitigs,self.mapGeneIdx[gene],self.margG[gene][g],g)
                os.remove(graphFileName) 
                os.remove(outFileName)
            self.addGamma(g)    
        print("-1,"+ str(self.div())) 
        self.updateGammaDelta()

    def exp_square_lambda(self):
        ''' Compute: sum_s E_q(phi,gamma) [ sum ( Phi_v Gamma_s )^2 ]. '''
        
        eLambda2Sum = self.eLambda*self.eLambda
        
        diagonal = np.dot(self.expPhi*self.expPhi,self.expGamma*self.expGamma)
        
        return np.sum(eLambda2Sum - diagonal + np.dot(self.expPhi2,self.expGamma2), axis = 1)

    def exp_square_lambda_matrix(self):
        ''' Compute: sum_s E_q(phi,gamma) [ sum ( Phi_v Gamma_s )^2 ]. '''
        
        eLambda2Sum = self.eLambda*self.eLambda
        
        diagonal = np.dot(self.expPhi*self.expPhi,self.expGamma*self.expGamma)
        
        return eLambda2Sum - diagonal + np.dot(self.expPhi2,self.expGamma2)


    def gene_mean_diff(self):
    
        diff_matrix = self.divF_matrix()
        gene_vals = defaultdict(list)
        
        for gene in self.genes:
            unitigs = self.mapUnitigs[gene]
            
            for unitig in unitigs:
                v_idx = self.mapGeneIdx[gene][unitig]
                
                gene_vals[gene].append(np.mean(diff_matrix[v_idx,:]))
            
        gene_means = {}
        
        for gene in self.genes:
            gene_means[gene] = np.mean(np.array(gene_vals[gene]))
                
        return gene_means



    def calc_elbo(self):
        ''' Compute the ELBO. '''
        total_elbo = 0.
        logLike = 0.
        exp_prior = 0.
        exp_q = 0.
        
        self.eLambda = np.dot(self.expPhi, self.expGamma) #maybe not necessary?
        
        logLike = -np.sum(self.eLambda*self.expTheta[:,np.newaxis])
        
        self.updateP() #also perhaps not necessary
        
        gammaDash = np.zeros((self.G + 1,self.S))
        
        gammaDash[0,:] = self.expLogDelta
        
        gammaDash[1,:] = self.expLogGamma
        
        logPhiDash = np.zeros((self.V,self.G +1))
        phiDash[:,1:] = self.expPhi
        
        logLike += np.einsum('vs,vsg,gs,vg',self.X,self.norm_p,gammaDash,phiDash)
        
        logLike -= np.sum(sp.special.gammaln(self.X + 1))
        
        exp_prior = 0.0
        # Prior lambdak, if using ARD, and prior U, V
        if self.ARD:
            
            exp_prior += self.alpha0 * math.log(self.beta0) - sp.special.gammaln(self.alpha0) \
                          + (self.alpha0 - 1.)*self.exp_loglambdak.sum() - self.beta0 * self.exp_lambdak.sum()
            
            exp_prior += self.S * np.log(self.exp_lambdak).sum() - (self.exp_lambdak[:,np.newaxis] * self.expGamma).sum()
            
        else:
            exp_prior += np.sum(-math.log(self.epsilon) - self.expGamma/self.epsilon)        
        
        #Prior theta if using bias
        
        if self.BIAS:
            #gamma prior on theta
            exp_prior += self.V*(self.alphaTheta0 * math.log(self.betaTheta0) - sp.special.gammaln(self.alphaTheta0))
        
            exp_prior += (self.alphaTheta0 - 1)*self.expLogTheta.sum() - self.alphaBeta0*self.expTheta.sum()
        
        #add phio prior assuming uniform 
        exp_prior += self.G*np.sum(self.logPhiPrior)
       
        #add prior for delta
        exp_prior +=   self.alphaDelta0 * math.log(self.betaDelta0) - sp.special.gammaln(self.alphaDelta0) 
        exp_prior +=  (self.alphaDelta0 - 1.)*self.expLogDelta - self.betaDelta0 * self.expDelta
        
        # q for lambdak, if using ARD ***check ****
        if self.ARD:
            exp_q +=  sum([v1*math.log(v2) for v1,v2 in zip(self.alphak_s,self.betak_s)]) - sum([sp.special.gammaln(v) for v in self.alphak_s]) \
                          + ((self.alphak_s - 1.)*self.exp_loglambdak).sum() - (self.betak_s * self.exp_lambdak).sum()        
        
        #q for gamma
        exp_q += np.sum(self.aGamma*np.log(self.bGamma)) - np.sum(sp.special.gammaln(self.aGamma)) \
                + np.sum((self.aGamma - 1.)*self.expLogGamma) - np.sum(self.bGamma*self.expGamma)
        
        #q for delta
        exp_q += self.aDelta*math.log(self.bDelta) - sp.special.gammaln(self.aDelta) + (self.aDelta - 1.)*self.expLogDelta - self.bDelta*self.expDelta
       
        #q for theta 
        if self.BIAS:
            exp_q += self.aTheta*math.log(self.bTheta) - sp.special.gammaln(self.aTheta) + (self.aTheta - 1.)*self.expLogTheta - self.bTheta*self.expTheta
        
        total_elbo = logLike + exp_prior - exp_q
        
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


    def calcPathDist(self, relax_path):
        
        dist = np.zeros((self.G,self.G))
        
        self.MAPs = defaultdict(list)
        
        pathsg = defaultdict(dict)
        
        for g in range(self.G):
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.assemblyGraphs[gene].unitigs
                self.updateUnitigFactorsMarg(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], self.margG[gene][g])

                factorGraph.reset()

                if not relax_path: 
                    factorGraph.var['zero+source+'].condition(1)

                    factorGraph.var['sink+infty+'].condition(1)
                else:
                    factorGraph.var['zero+source+'].clear_condition()

                    factorGraph.var['sink+infty+'].clear_condition()
                
                maximals = self.runFGMaximal(factorGraph, g)
                
                self.MAPs[gene].append(maximals)
                biGraph = self.factorDiGraphs[gene]

                pathG = self.convertMAPToPath(self.MAPs[gene][g],biGraph)
                pathG.pop(0)
                pathsg[g][gene] = pathG
        
        for g in range(self.G):
            for h in range(g+1,self.G):
                diff = 0
                comp = 0
                for gene in self.genes:
                    if len(set(pathsg[g][gene])) > 0 and len(set(pathsg[h][gene])) > 0:
                        comp += 1 
                        diff += len(set(pathsg[g][gene]) ^ set(pathsg[h][gene]))
                dist[g,h] = diff     

        return dist
        
    ''' Removes strain below a given total intensity and degenerate'''
    def removeRedundant(self, minIntensity, gammaIter, relax_path):
    
        #calculate number of good strains
        nNewG = 0
        
        sumIntensity = np.max(self.expGamma,axis=1)
        
        dist = self.calcPathDist(relax_path)
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
            newLogPhi = self.expLogPhi[:,retained]
            
            newExpGamma  = self.expGamma[retained,:]
            newExpGamma2 = self.expGamma2[retained,:]
            newAGamma    = self.aGamma[retained,:] 
            newBGamma    = self.bGamma[retained,:]
            newLogGamma  = self.expLogGamma[retained,:]
            newVarGamma  = self.varGamma[retained,:]
        
            self.expPhi  = newPhi
            self.expPhi2 = newPhi2
            self.HPhi    = newPhiH
            self.expLogPhi = newLogPhi
        
            self.expGamma  = newExpGamma
            self.expGamma2 = newExpGamma2
            self.varGamma  = newVarGamma
            self.expLogGamma = newLogGamma
            
            self.aGamma = newAGamma
            self.bGamma = newBGamma
        
            self.norm_p = np.zeros((self.V, self.S,self.G + 1))

            if self.ARD:
                self.alphak_s = self.alphak_s[retained]
                self.betak_s  = self.betak_s[retained]
                self.exp_lambdak = self.exp_lambdak[retained]
                self.exp_loglambdak = self.exp_loglambdak[retained]
        
        
            iter = 0    
            while iter < gammaIter:
        
                if self.ARD:
                    for g in range(self.G):
                        self.update_lambdak(g)
                        self.update_exp_lambdak(g)
                
                self.updateGammaDelta(g)

                self.eLambda = np.dot(self.expPhi,self.expGamma)
            
                self.updateTheta()
            
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

                maximals = self.runFGMaximal(factorGraph, g)

                self.MAPs[gene].append(maximals)

                pathG = self.convertMAPToPath(self.MAPs[gene][g],biGraph)
                pathG.pop(0)
                pathsg[g][gene] = pathG
        for g in range(self.G):
            for h in range(g+1,self.G):
                diff = 0
                for gene in self.genes:
                    if len(set(pathsg[g][gene])) > 0 and len(set(pathsg[h][gene])) > 0:
                        diff += len(set(pathsg[g][gene]) ^ set(pathsg[h][gene]))
                dist[g,h] = diff 
        
        
        collapsed = [False]*self.G 
        for g in range(self.G):
            
            for h in range(g+1,self.G):
                if dist[g,h] == 0 and collapsed[h] == False:
                    self.expGamma[g,:] += self.expGamma[h,:]
                    self.expGamma[h,:] = 0.
                    self.expLogGamma[h,:] = 0.
                    
                    collapsed[h] = True

    def writeMarginals(self,fileName):

        with open(fileName, "w") as margFile:
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.mapUnitigs[gene]

                for unitig in unitigs:
                    v_idx = unitig_drops = self.mapGeneIdx[gene][unitig]
                    vals = [str(x) for x in self.expPhi[v_idx,:].tolist()]
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


    def writeMaximals(self,fileName,drop_strain=None):

        if drop_strain is None:
            drop_strain = {gene:[False]*self.G for gene in self.genes}

        with open(fileName, "w") as margFile:
            for gene, factorGraph in self.factorGraphs.items():
                unitigs = self.assemblyGraphs[gene].unitigs

                for unitig in unitigs:
                    if unitig in self.margG[gene][0]:
                        vals = []
                        for g in range(self.G):
                            if not drop_strain[gene][g]:
                                if self.MAPs[gene][g][unitig] == 1:
                                    vals.append(g)                  
                        vString = "\t".join([str(x) for x in vals])

                        margFile.write(gene + "_" + unitig + "\t" + vString + "\n")

    def runFGMaximal(self, factorGraph, g):
        graphString = str(factorGraph)
                    
        outFileStub = str(uuid.uuid4()) + 'graph_'+ str(g)
        graphFileName = self.working_dir + '/' + outFileStub + '.fg'                
        outFileName = self.working_dir + '/' + outFileStub + ".out"

        with open(graphFileName, "w") as text_file:
            print(graphString, file=text_file)


        cmd = self.fgExePath + 'runfg_flex ' + graphFileName + ' ' + outFileName + ' 1 -1' 

        p = Popen(cmd, stdout=PIPE,shell=True)
             
        p.wait()

        with open (outFileName, "r") as myfile:
            outString=myfile.readlines()
       
        os.remove(graphFileName)
        os.remove(outFileName)
        
        return self.parseFGString(factorGraph, outString)
                    


    def getMaximalUnitigs(self,fileName,drop_strain=None,relax_path=False):

        if drop_strain is None:
            drop_strain = {gene:[False]*self.G for gene in self.genes}

        self.MAPs = defaultdict(list)
        haplotypes = defaultdict(list)

        output = np.sum(self.expGamma,axis=1) > self.minIntensity

        for g in range(self.G):
            for gene, factorGraph in self.factorGraphs.items():
                if not drop_strain[gene][g]:
        
                    unitigs = self.assemblyGraphs[gene].unitigs

                    self.updateUnitigFactorsMarg(unitigs, self.mapGeneIdx[gene], self.unitigFactorNodes[gene], self.margG[gene][g])        
                    factorGraph.reset()

                    if not relax_path:
                        factorGraph.var['zero+source+'].condition(1)

                        factorGraph.var['sink+infty+'].condition(1)
                    else:
                        factorGraph.var['zero+source+'].clear_condition()

                        factorGraph.var['sink+infty+'].clear_condition()
                    
                    maximals = self.runFGMaximal(factorGraph, g)
                    
                    self.MAPs[gene].append(maximals) 
                    
                    biGraph = self.factorDiGraphs[gene]
                
                    pathG = self.convertMAPToPath(self.MAPs[gene][g],biGraph)
                    pathG.pop(0)
                    if len(pathG) > 0:
                        unitig = self.assemblyGraphs[gene].getUnitigWalk(pathG)
                        haplotypes[gene].append(unitig)
                    else:
                        haplotypes[gene].append("")
                        
                else:
                    haplotypes[gene].append("")
                    self.MAPs[gene].append(None)            
                    
        with open(fileName, "w") as fastaFile:
            for g in range(self.G):
                if output[g]:
                    for gene, factorGraph in self.factorGraphs.items():
                        if not drop_strain[gene][g] and len(haplotypes[gene][g]) > 0:
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

                maximals = self.runFGMaximal(factorGraph, r)
                    
                refMAPs[r] = maximals
            
                self.margG[gene][ref] = self.convertMAPMarg(refMAPs[r],factorGraph.mapNodes)
                self.updateExpPhi(unitigs,self.mapGeneIdx[gene],self.margG[gene][ref],r)
        
                biGraph = self.factorDiGraphs[gene]
                pathG = self.convertMAPToPath(refMAPs[r], biGraph)
                pathG.pop(0)
                unitig = self.assemblyGraphs[gene].getUnitigWalk(pathG)
                    
                refs.append(unitig)    
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

    parser.add_argument('--no_ARD', dest='ARD', default=True, action='store_false')

    parser.add_argument('--no_BIAS', dest='BIAS', default=True, action='store_false')

    args = parser.parse_args()

    import ipdb; ipdb.set_trace()

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

    assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G = args.strain_number, readLength=args.read_length, ARD=args.ARD,BIAS=args.BIAS)
    
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

        assGraph.update(200, True, logFile=None,drop_strain=None,relax_path=True)
        
        gene_mean_error = assGraph.gene_mean_diff()
        
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


