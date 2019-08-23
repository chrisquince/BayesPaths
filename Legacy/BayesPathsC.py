import argparse
import sys
import glob
import numpy as np
import os
import re

from GraphProcess import getMaximumCoverageWalk
from Utils.UnitigGraph import UnitigGraph
from AssemblyPathSVAB import AssemblyPathSVA
from Utils.UtilsFunctions import convertNodeToName
from numpy.random import RandomState

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("Gene_dir", help="directory with gfa files in")

    parser.add_argument("kmer_length", help="kmer length assumed overlap")

    parser.add_argument("outFileStub", help="output file stub")

    parser.add_argument('-g','--strain_number',nargs='?', default=5, type=int, 
        help=("maximum number of strains"))

    parser.add_argument('-f','--frac',nargs='?', default=0.75, type=float, 
        help=("fraction for path source sink"))

    parser.add_argument('-r','--readLength',nargs='?', default=100., type=float,
        help=("read length used for sequencing defaults 100bp"))

    parser.add_argument('-s', '--random_seed', default=23724839, type=int,
        help="specifies seed for numpy random number generator defaults to 23724839 applied after random filtering")


    args = parser.parse_args()

    import ipdb; ipdb.set_trace()
    
    np.random.seed(args.random_seed) #set numpy random seed not needed hopefully
    prng = RandomState(args.random_seed) #create prng from seed 
    
    gfaFiles = glob.glob(args.Gene_dir + '/*.gfa')    

    assemblyGraphs = {} #dictionary of assembly graphs by gene name
    sink_maps = {} # sinks (in future these defined outside)
    source_maps = {} #sources
    cov_maps = {} #coverages
    geneOrder = []
    for gfaFile in gfaFiles:
        fileName = os.path.basename(gfaFile)

        p = re.compile('COG[0-9]+')

        m = p.search(gfaFile)
        
        if m is None:
            raise ValueError

        gene = m.group()
        geneOrder.append(gene)
        covFile = gfaFile[:-3] + "csv"
        
        unitigGraph = UnitigGraph.loadGraphFromGfaFile(gfaFile,int(args.kmer_length), covFile)
            
        (source_list, sink_list) = unitigGraph.selectSourceSinks(args.frac)

        source_names = [convertNodeToName(source) for source in source_list] 
        sink_names = [convertNodeToName(sink) for sink in sink_list]
            
        sink_maps[gene] = sink_list
        source_maps[gene] = source_list
        assemblyGraphs[gene] = unitigGraph
    
    combinedGraph = UnitigGraph.combineGraphs(geneOrder, assemblyGraphs, source_maps, sink_maps)
    combinedGraph.writeToGFA("Combined.gfa")
    
    newGraphs = {'combined':combinedGraph}
    newSourceMaps = {'combined':source_maps[geneOrder[0]]}
    newSinkMaps = {'combined':sink_maps[geneOrder[-1]]}    

    assGraph = AssemblyPathSVA(prng,newGraphs,newSourceMaps, newSinkMaps, G = args.strain_number, readLength=args.readLength,ARD=True,BIAS=True)
    
    assGraph.initNMF()

    assGraph.update(100, True)
        
    assGraph.writeMarginals(args.outFileStub + "margFile.csv")
   
    assGraph.getMaximalUnitigs(args.outFileStub + "Haplo_" + str(assGraph.G) + ".fa")
 
    assGraph.writeMaximals(args.outFileStub + "maxFile.tsv")
   
    assGraph.writeGammaMatrix(args.outFileStub + "Gamma.csv") 

    assGraph.writeTheta(args.outFileStub + "Theta.csv") 

if __name__ == "__main__":
    main(sys.argv[1:])
