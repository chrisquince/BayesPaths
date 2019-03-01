import argparse
import sys
import glob
import numpy as np
import os
import re

from GraphProcess import getMaximumCoverageWalk
from Utils.UnitigGraph import UnitigGraph
from AssemblyPath.AssemblyPathSVAF import AssemblyPathSVA
from Utils.UtilsFunctions import convertNodeToName
from numpy.random import RandomState

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("Gene_dir", help="directory with gfa files in")

    parser.add_argument("kmer_length", help="kmer length assumed overlap")

    parser.add_argument("outFileStub", help="output file stub")

    parser.add_argument('-g','--strain_number',nargs='?', default=5, type=int, 
        help=("maximum number of strains"))

    parser.add_argument('-n','--ncat',nargs='?', default=1, type=int, 
        help=("number of noise categories"))

    parser.add_argument('-r','--readLength',nargs='?', default=100., type=float,
        help=("read length used for sequencing defaults 100bp"))

    parser.add_argument('-s', '--random_seed', default=23724839, type=int,
        help="specifies seed for numpy random number generator defaults to 23724839 applied after random filtering")

    parser.add_argument('-e','--executable_path',nargs='?', default='./runfg_source/', type=str,
        help=("path to factor graph executable"))

    args = parser.parse_args()

    #import ipdb; ipdb.set_trace()
    
    np.random.seed(args.random_seed) #set numpy random seed not needed hopefully
    prng = RandomState(args.random_seed) #create prng from seed 
    
    gfaFiles = glob.glob(args.Gene_dir + '/*.gfa')    

    assemblyGraphs = {} #dictionary of assembly graphs by gene name
    sink_maps = {} # sinks (in future these defined outside)
    source_maps = {} #sources
    cov_maps = {} #coverages
    gfaFiles.sort()
    for gfaFile in gfaFiles:
        fileName = os.path.basename(gfaFile)

        p = re.compile('COG[0-9]+')

        m = p.search(gfaFile)
        
        if m is None:
            raise ValueError

        gene = m.group()
        
        covFile = gfaFile[:-3] + "csv"
        
        unitigGraph = UnitigGraph.loadGraphFromGfaFile(gfaFile,int(args.kmer_length), covFile)
            
        deadEndFile = gfaFile[:-3] + "deadends"
        
        stopFile = gfaFile[:-3] + "stops"
        
        deadEnds = []
        with open(deadEndFile) as f:
            for line in f:
                line.strip()
                deadEnds.append(line)
        
        stops = []
        
        with open(stopFile) as f:
            for line in f:
                line = line.strip()
                toks = line.split("\t")
                dirn = True
                if toks[1] == '-':
                    dirn = False
                stops.append((toks[0],dirn))
        
        (source_list, sink_list) = unitigGraph.selectSourceSinksStops(stops, deadEnds)
        

        source_names = [convertNodeToName(source) for source in source_list] 
        sink_names = [convertNodeToName(sink) for sink in sink_list]
            
        sink_maps[gene] = sink_list
        source_maps[gene] = source_list
        assemblyGraphs[gene] = unitigGraph
    
    
    assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G = args.strain_number, readLength=args.readLength,ARD=True,BIAS=True, fgExePath=args.executable_path,nTauCats=args.ncat)
    
    assGraph.initNMF()

    assGraph.update(200, True,logFile=args.outFileStub + "_log.txt",drop_strain=None,relax_path=True)
 
    #gene_mean_error = assGraph.gene_mean_diff()

    #strain_drop_elbo = assGraph.calc_strain_drop_elbo()
            
    #for (gene,drops) in strain_drop_elbo.items():
    
     #   for (g, drop) in enumerate(drops):
      #      if drop:
       #         assGraph.drop_gene_strains(gene, g)
                
     
            
    #assGraph.update(100,False,logFile=args.outFileStub + "_log.txt",drop_strain=strain_drop_elbo)

    assGraph.writeMarginals(args.outFileStub + "margFile.csv")
   
    assGraph.getMaximalUnitigs(args.outFileStub + "Haplo_" + str(assGraph.G) + ".fa",drop_strain=None, relax_path=True)
 
    assGraph.writeMaximals(args.outFileStub + "maxFile.tsv",drop_strain=None)
   
    assGraph.writeGammaMatrix(args.outFileStub + "Gamma.csv") 

    assGraph.writeTheta(args.outFileStub + "Theta.csv") 

if __name__ == "__main__":
    main(sys.argv[1:])
