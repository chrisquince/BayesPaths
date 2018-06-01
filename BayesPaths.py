import argparse
import sys
import glob
import numpy as np
import os

from UnitigGraph import UnitigGraph
from AssemblyPathSVA import AssemblyPathSVA
from Utils import convertNodeToName
from numpy.random import RandomState

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("Gene_dir", help="directory with gfa files in")

    parser.add_argument("cov_file", help="coverage file")

    parser.add_argument("kmer_length", help="kmer length assumed overlap")

    parser.add_argument('-g','--strain_number',nargs='?', default=5, type=int, 
        help=("maximum number of strains"))

    parser.add_argument('-f','--frac',nargs='?', default=0.75, type=float, 
        help=("fraction for path source sink"))

    args = parser.parse_args()

    import ipdb; ipdb.set_trace()
    
    np.random.seed(2)
    prng = RandomState(238329)
    
    gfaFiles = glob.glob(args.Gene_dir + '/*.gfa')    

    assemblyGraphs = {}
    sink_maps = {}
    source_maps = {}
    
    for gfaFile in gfaFiles:
        fileName = os.path.basename(gfaFile)
        
        gene = fileName.split('_')[0]
    
        unitigGraph = UnitigGraph.loadGraphFromGfaFile(gfaFile,int(args.kmer_length), args.cov_file)
            
        (source_list, sink_list) = unitigGraph.selectSourceSinks2(args.frac)

        source_names = [convertNodeToName(source) for source in source_list] 
        sink_names = [convertNodeToName(sink) for sink in sink_list]
            
        sink_maps[gene] = sink_list
        source_maps[gene] = source_list
        assemblyGraphs[gene] = unitigGraph

    assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G = args.strain_number, readLength=150,ARD=True)
    
    assGraph.initNMF()

    assGraph.update(50)
        
    assGraph.getMaximalUnitigs("Haplo_" + str(assGraph.G) + ".fa")

if __name__ == "__main__":
    main(sys.argv[1:])