import argparse
import sys
from BayesPaths.UnitigGraph import UnitigGraph
from BayesPaths.UtilsFunctions import convertNodeToName
from collections import defaultdict
import networkx as nx


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("gfa_file", help="assembly graph in gfa format")
    
    parser.add_argument("kmer_length", help="kmer length assumed overlap")
    
    parser.add_argument("cov_file", help="coverages")

    parser.add_argument("list_mags", help="list of mags")
    
    parser.add_argument("unitig_ass", help="unitig bin ass")

    parser.add_argument("out_stub", help="output file stub")
    
    args = parser.parse_args()

    import ipdb; ipdb.set_trace()
    
    with open(args.list_mags,'r') as cog_file:
        mags = {line.rstrip() for line in cog_file}


    unitig_ass = defaultdict(lambda: -1)
    with open(args.unitig_ass,'r') as cog_file:
        for line in cog_file:
            line = line.rstrip()
            
            toks = line.split(',')
            
            unitig_ass[toks[0]] = int(toks[2])

    unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.gfa_file,int(args.kmer_length), args.cov_file)

    for unitig in ['2297102']:
    
        if unitig_ass[unitig] not in mags:
            print('Debug')

            fNode = unitig + '+'
            for succ in nx.bfs_tree(unitigGraph.directedUnitigBiGraph, fNode, 10):
            
                ass_bin = unitig_ass[succ[:-1]]

    

    
if __name__ == "__main__":
    main(sys.argv[1:])
