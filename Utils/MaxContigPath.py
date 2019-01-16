import argparse
import glob, sys, os
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
    
    parser.add_argument("path_file_dir", help="directory of paths")
    
    #parser.add_argument("cov_file", help="coverages")
    
    parser.add_argument("out_stub", help="output_stub")
     
    args = parser.parse_args()

    import ipdb; ipdb.set_trace()

    #unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.gfa_file,int(args.kmer_length), args.cov_file)

    unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.gfa_file,int(args.kmer_length))

    os.chdir(args.path_file_dir)
    contigs = {}
    filenames = {}
    for file in glob.glob("*.txt"):
        print(file)
        #NODE_469_length_15023_cov_447.352084_10 COG0016 strand=+
        
        with open(file) as txt:
            contig_namet = txt.readline()
            contig_namet =  contig_namet.rstrip()
            path = txt.readline()
            path_string = path.rstrip()
            path_nodest = path_string.split(',')
            contigs[contig_namet] = path_nodest
            filenames[contig_namet] = file
    
    new_seqs = {}
    for contig_name, path_nodes in contigs.items():

        bContig = True
        
        dGraph = unitigGraph.directedUnitigBiGraph
        
        for node1,node2 in zip(path_nodes,path_nodes[1:]):
            if not dGraph.has_edge(node1, node2):
                bContig = False
                break

        if not bContig:
            print("Not contigious")
            hashDist = {x:1.0 for x in path}
    
        
            for (u,v) in dGraph.edges:
                if u in hashDist:
                    dGraph.edges[u,v]['weight'] = 0.
                else:
                    dGraph.edges[u,v]['weight'] = 1.
    
            source = path_nodes[0]
            sink = path_nodes[-1]
    
            best_path = nx.shortest_path(dGraph, source=source, target=sink, weight='weight')
    
            for node in best_path:
                print(node)
    
            seq = unitigGraph.getUnitigWalk(best_path)
            new_seqs[contig_name] = seq
        
            newfilename = filenames[contig_namet][:-4] + "_c.fna"
        
            with open(newfilename, 'w') as out:
                out.write(">" + contig_name + "\n")
                out.write(seq + "\n")
        else:
            seq = unitigGraph.getUnitigWalk(path_nodes)
            new_seqs[contig_name] = seq
        
            newfilename = filenames[contig_namet][:-4] + ".fna"
        
            with open(newfilename, 'w') as out:
                out.write(">" + contig_name + "\n")
                out.write(seq + "\n")
        
            
    
if __name__ == "__main__":
    main(sys.argv[1:])
