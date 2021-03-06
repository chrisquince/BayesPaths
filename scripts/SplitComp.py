import argparse
import sys
from UnitigGraph import UnitigGraph
from UtilsFunctions import convertNodeToName
import networkx as nx


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("gfa_file", help="assembly graph in gfa format")
    
    parser.add_argument("kmer_length", help="kmer length assumed overlap")
    
    parser.add_argument("cov_file", help="coverages")

    parser.add_argument("out_stub", help="output file stub")
    
    args = parser.parse_args()

    #import ipdb; ipdb.set_trace()

    unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.gfa_file,int(args.kmer_length), args.cov_file)

    components = sorted(nx.connected_components(unitigGraph.undirectedUnitigGraph), key = len, reverse=True)

    c = 0
    for component in components:
        unitigSubGraph = unitigGraph.createUndirectedGraphSubset(component)
        
        unitigSubGraph.writeToGFA(args.out_stub + 'component_' + str(c) + '.gfa')

        unitigSubGraph.writeCovToCSV(args.out_stub + 'component_' + str(c) + '.csv')
        
        c = c + 1

    
if __name__ == "__main__":
    main(sys.argv[1:])
