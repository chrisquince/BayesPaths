import argparse
import sys
from BayesPaths.UnitigGraph import UnitigGraph
from BayesPaths.UtilsFunctions import convertNodeToName
from collections import defaultdict
from collections import Counter
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
        mags = {int(line.rstrip()) for line in cog_file}


    color_map = {}
    unitig_ass = defaultdict(lambda: -1)
    with open(args.unitig_ass,'r') as cog_file:
        for line in cog_file:
            line = line.rstrip()
            
            toks = line.split(',')
            
            unitig_ass[toks[0]] = int(toks[2])
            if int(toks[2]) not in color_map: 
                color_map[int(toks[2])] = toks[1] 
    unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.gfa_file,int(args.kmer_length), args.cov_file)

    unitigNew = {}
    magAss = defaultdict(set)
    for unitig in unitigGraph.unitigs:
    
        if unitig_ass[unitig] not in mags:
            

            fNode = unitig + '+'
            binCounts = Counter()
            total = 0.
            for succ in nx.bfs_successors(unitigGraph.directedUnitigBiGraph, fNode, 20):
                
                sUnitig = succ[0][:-1]
                ass_bin = unitig_ass[sUnitig]
                
                if ass_bin in mags:
                    total += unitigGraph.lengths[sUnitig]
                    binCounts[ass_bin] += unitigGraph.lengths[sUnitig]

            rNode = unitig + '-'
            for succ in nx.bfs_successors(unitigGraph.directedUnitigBiGraph, rNode, 20):
                sUnitig = succ[0][:-1]
                ass_bin = unitig_ass[sUnitig]

                if ass_bin in mags:
                    total += unitigGraph.lengths[sUnitig]
                    binCounts[ass_bin] += unitigGraph.lengths[sUnitig]


            if total > 0.:
                new_ass = binCounts.most_common()[0][0]
        
                new_color = color_map[new_ass]
                unitigNew[unitig] = new_ass
                magAss[new_ass].add(unitig)
                #print(unitig + "," + new_color + "," + str(new_ass))    
        else:
            mag = unitig_ass[unitig]

            new_color = color_map[mag]
            unitigNew[unitig] = mag
            magAss[mag].add(unitig)
            #print(unitig + "," + new_color + "," + str(mag))
    
    graph14 = unitigGraph.createUndirectedGraphSubset(list(magAss[14]))
    graph14.writeToGFA('graph14.gfa')


if __name__ == "__main__":
    main(sys.argv[1:])
