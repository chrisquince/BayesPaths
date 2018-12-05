import argparse
import sys
from UnitigGraph import UnitigGraph
from Utils import convertNodeToName
import networkx as nx
import numpy as np

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("gfa_file", help="assembly graph in gfa format")
    
    parser.add_argument("kmer_length", help="kmer length assumed overlap")
    
    parser.add_argument("cov_file", help="coverages")
    
    parser.add_argument("out_stub", help="output_stub")
    
    parser.add_argument('-k', '--kAbund', action='store_true',help=("input coverages as kmer counts"))
    
    args = parser.parse_args()

    import ipdb; ipdb.set_trace()

    unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.gfa_file,int(args.kmer_length), args.cov_file)

    components = sorted(nx.connected_components(unitigGraph.undirectedUnitigGraph), key = len, reverse=True)
    #probably haves separate components but no matter
    c = 0
    for component in components:
        unitigSubGraph = unitigGraph.createUndirectedGraphSubset(component)
        
        if nx.is_directed_acyclic_graph(unitigSubGraph.directedUnitigBiGraph) is True:
            dGraph = unitigSubGraph.directedUnitigBiGraph
            
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
                
                if args.kAbund == True:
                    lengthPlus = 1.
                else:
                    if unitigSubGraph.directedUnitigBiGraph.out_degree(node) == 0:
                        lengthPlus = unitigSubGraph.lengths[noded]
                    else:
                        lengthPlus = unitigSubGraph.lengths[noded] - unitigSubGraph.overlapLength
                
                myWeight = np.sum(unitigSubGraph.covMap[noded])*lengthPlus
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
            
            for node in minPath:
                print(node[:-1])
            
            #print("Dummy")
                            
            maxSeq = unitigSubGraph.getUnitigWalk(minPath)
        
            with open(args.out_stub + ".fa", "w") as fastaFile:
                fastaFile.write(">" + args.out_stub + "\n")
                fastaFile.write(maxSeq + "\n")

            covPath = unitigSubGraph.computePathCoverage(minPath)

            with open(args.out_stub + ".csv", "w") as covFile:
                cList = covPath.tolist()
                cString = ",".join([str(x) for x in cList])
                covFile.write(cString  + "\n") 
        else:
            print("Component " + str(c) + " is not a DAG")
            
        c = c + 1


    
if __name__ == "__main__":
    main(sys.argv[1:])
