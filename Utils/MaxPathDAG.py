import argparse
import sys
from UnitigGraph import UnitigGraph
from UtilsFunctions import convertNodeToName
import networkx as nx
import numpy as np
from collections import deque

def removeCycles3(directedGraph1,weight):
    directedGraph2 = directedGraph1.copy()

    cycles_removed = 0
    while True:
    
        try:
            cycle = nx.find_cycle(directedGraph2)
            
            min_link = sys.float_info.max 
            min_edge = None
            
            for edge in cycle:
                edge_weight = directedGraph2[edge[0]][edge[1]][weight]
                if min_link > edge_weight:
                    min_link = edge_weight
                    min_edge = edge
            
            print("Removed cycle " + str(cycles_removed) + " " + min_edge[0] + "->" + min_edge[1] + " =" + str(min_link))            
            directedGraph2.remove_edge(min_edge[0],min_edge[1])
            
            cycles_removed +=1
            
        except nx.exception.NetworkXNoCycle:
            break
    
    return directedGraph2
    
def removeCycles2(directedGraph1, weight):
    directedGraph2 = directedGraph1.copy()
    
    simple_cycles = list(nx.simple_cycles(directedGraph2))
    cycles_removed = 0
    while len(simple_cycles) > 0:
    
        for cycle in simple_cycles:
            min_link = sys.float_info.max 
            min_edge = None
                        
            cycle.append(cycle[0])
            
            for (current_node, next_node) in zip(list(cycle),list(cycle)[1:]):
                if next_node in directedGraph2[current_node]:
                    if min_link > directedGraph2[current_node][next_node][weight]:
                        min_link = directedGraph2[current_node][next_node][weight]
                        min_edge = (current_node,next_node)
                else:
                    min_link = -1.
                    break
            
            if min_link >= 0.0:
                directedGraph2.remove_edge(min_edge[0],min_edge[1])
            
            cycles_removed +=1
    
        simple_cycles = list(nx.simple_cycles(directedGraph2))
    
    return directedGraph2

def removeCycles(directedGraph1, weight):
    directedGraph2 = directedGraph1.copy()

    nNodes = directedGraph2.number_of_nodes()
    
    color = {x:0 for x in directedGraph2.nodes}
    #INTM* pB = G2.pB();
    #INTM* r = G2.r();
    #T* v = G2.v();

    cycle_detected=True
    cycles_removed=0;
    cycles_toremove = []
   
    while cycle_detected:
        cycle_detected=False 
        
        node_list = deque()
        
        for node in directedGraph2.nodes:
            if color[node] != 2:
                color[node] = 0
                node_list.append(node)

        current_path = deque()
        while node_list:
        
            node = node_list.popleft()
            
            if color[node] == 0:
                current_path.appendleft(node)
                color[node] = 1
                
                for child in directedGraph2[node]:
                    if color[child] == 1:
                        cycle_detected=True
                        reverse_path = current_path.copy()
                        reverse_path.reverse()
                        
                        while True: 
                            if reverse_path[0] == child:
                                break
                            reverse_path.popleft()
                        
                        reverse_path.append(child)
                        
                        min_link = sys.float_info.max
                        
                        min_node = -1
                        
                        for (current_node, next_node) in zip(list(reverse_path),list(reverse_path)[1:]):
                            if min_link > directedGraph2[current_node][next_node][weight]:
                                min_link = directedGraph2[current_node][next_node][weight]
                                min_edge = (current_node,next_node)
                        #v[min_node] = 0.
                        directedGraph2.remove_edge(min_edge[0],min_edge[1])
                        cycles_toremove.append((min_edge,min_link))
                        node_list = deque()
                        cycles_removed +=1
                        break
                    else:
                        node_list.appendleft(child)
            elif color[node] == 1:
                #means descendants(node) is acyclic
                color[node]=2
                node_list.popleft()
                current_path.popleft()
            elif color[node] == 2:
                node_list.popleft()
    
    return directedGraph2

def calcMaxPath(dGraph,unitigSubGraph,sampleList, kAbund):

    assert nx.is_directed_acyclic_graph(dGraph)

    top_sort = list(nx.topological_sort(dGraph))
    lenSort = len(top_sort)
            
    maxPred = {}
    maxWeightNode = {}
    for node in top_sort:
        maxWeight = 0.
        maxPred[node] = None
        noded = node[:-1]
        for predecessor in dGraph.predecessors(node):
            weight = maxWeightNode[predecessor]
            if weight > maxWeight:
                maxWeight = weight
                maxPred[node] = predecessor
                
        if kAbund == True:
            lengthPlus = 1.
        else:
            if dGraph.out_degree(node) == 0:
                lengthPlus = unitigSubGraph.lengths[noded]
            else:
                lengthPlus = unitigSubGraph.lengths[noded] - unitigSubGraph.overlapLength
                
        myWeight = np.sum(unitigSubGraph.covMap[noded][sampleList])*lengthPlus
        maxWeightNode[node] = maxWeight + myWeight
                
                
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
                            
    maxSeq = unitigSubGraph.getUnitigWalk(minPath)
    covPath = unitigSubGraph.computePathCoverage(minPath)

    return (minPath, maxSeq, covPath)

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("gfa_file", help="assembly graph in gfa format")
    
    parser.add_argument("kmer_length", help="kmer length assumed overlap")
    
    parser.add_argument("cov_file", help="coverages")
    
    parser.add_argument("out_stub", help="output_stub")
    
    parser.add_argument('-k', '--kAbund', action='store_true',help=("input coverages as kmer counts"))
    
    args = parser.parse_args()

    #import ipdb; ipdb.set_trace()

    unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.gfa_file,int(args.kmer_length), args.cov_file)

    components = sorted(nx.connected_components(unitigGraph.undirectedUnitigGraph), key = len, reverse=True)
    #probably haves separate components but no matter
    c = 0
    for component in components:
        unitigSubGraph = unitigGraph.createUndirectedGraphSubset(component)

        dGraph = unitigSubGraph.directedUnitigBiGraph
        
        if nx.is_directed_acyclic_graph(dGraph) is not True:
            print("Component " + str(c) + " is not a DAG")
            dGraph = removeCycles3(dGraph, 'covweight')
            print("Component " + str(c) + " is now a DAG")
        
        nS = unitigSubGraph.covMap[unitigSubGraph.unitigs[0]].shape[0]
        (minPath, maxSeq, covPath) = calcMaxPath(dGraph,unitigSubGraph,range(nS), args.kAbund)
        outc = args.out_stub + "_" + str(c)
        with open(outc + ".tsv", "w") as tsvFile:
            for node in minPath:
                tsvFile.write(node + "\n")

        with open(outc + ".fa", "w") as fastaFile:
            fastaFile.write(">" + args.out_stub + "\n")
            fastaFile.write(maxSeq + "\n")

        with open(outc + ".csv", "w") as covFile:
            cList = covPath.tolist()
            cString = ",".join([str(x) for x in cList])
            covFile.write(cString  + "\n") 
            
        for s in range(nS):
                  
            (minPathS, maxSeqS, covPathS) = calcMaxPath(dGraph,unitigSubGraph,[s], args.kAbund)

            with open(outc + "_" + str(s) + ".tsv", "w") as tsvFile:
                for node in minPathS:
                    tsvFile.write(node + "\t" + str(s) + "\n")

            with open(outc + "_" + str(s) + ".fa", "w") as fastaFile:
                fastaFile.write(">" + args.out_stub + "\n")
                fastaFile.write(maxSeqS + "\n")

            with open(outc +  "_" + str(s) + ".csv", "w") as covFile:
                cList = covPathS.tolist()
                cString = ",".join([str(x) for x in cList])
                covFile.write(cString  + "\n")   
          
        c = c + 1


    
if __name__ == "__main__":
    main(sys.argv[1:])
