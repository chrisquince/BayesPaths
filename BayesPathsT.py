import argparse
import sys
import glob
import numpy as np
import os
import re

from GraphProcess import getMaximumCoverageWalk
from Utils.UnitigGraph import UnitigGraph
from AssemblyPath.AssemblyPathSVAG import AssemblyPathSVA
from Utils.UtilsFunctions import convertNodeToName
from numpy.random import RandomState

def filterGenes(assGraph):
    gene_mean_error = assGraph.gene_mean_diff()
    gene_mean_elbo = assGraph.gene_mean_elbo()

    errors = []
    genes = []
    
    for (gene, error) in gene_mean_error.items():
        print(gene + "," + str(error) + "," + str(gene_mean_elbo[gene]))
        errors.append(error)
        genes.append(gene)
    
    error_array = np.array(errors)
    medianErr = np.median(error_array)
    devArray = np.absolute(error_array - medianErr)
    medianDevError = np.median(devArray)

    genesSelect = []
    for gidx, gene in enumerate(genes):
        if devArray[gidx] > 3.0*medianDevError and error_array[gidx] > medianErr:
            print("Removing: " + str(gene))
        else:
            genesSelect.append(gene)

    return genesSelect 

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("Gene_dir", help="directory with gfa files in")

    parser.add_argument("kmer_length", help="kmer length assumed overlap")

    parser.add_argument("outFileStub", help="output file stub")

    parser.add_argument('-p','--frac',nargs='?', default=0.1, type=float, 
        help=("fraction for path source sink"))

    parser.add_argument('-f','--frac_cov',nargs='?', default=0.02, type=float, 
        help=("fractional coverage for noise nodes"))

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
        print(gene)        
        covFile = gfaFile[:-3] + "tsv"
        
        unitigGraph = UnitigGraph.loadGraphFromGfaFile(gfaFile,int(args.kmer_length), covFile, True)
            
        
        (source_list, sink_list) = unitigGraph.selectSourceSinks(args.frac)
        

        source_names = [convertNodeToName(source) for source in source_list] 
        sink_names = [convertNodeToName(sink) for sink in sink_list]
            
        sink_maps[gene] = sink_list
        source_maps[gene] = source_list
        assemblyGraphs[gene] = unitigGraph
    
    
    assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G = args.strain_number, readLength=args.readLength,ARD=True,BIAS=True, fgExePath=args.executable_path,nTauCats=args.ncat,fracCov = args.frac_cov)
    
    genesRemove = assGraph.get_outlier_cogs_sample(mCogFilter = 3.0, cogSampleFrac=0.80)
    
    genesFilter = list(set(assGraph.genes) ^ set(genesRemove))

    assemblyGraphsFilter = {s:assemblyGraphs[s] for s in genesFilter}
    source_maps_filter = {s:source_maps[s] for s in genesFilter} 
    sink_maps_filter = {s:sink_maps[s] for s in genesFilter}
    
    assGraph = AssemblyPathSVA(prng, assemblyGraphsFilter, source_maps_filter, sink_maps_filter, G = args.strain_number, readLength=args.readLength,ARD=True,BIAS=True, fgExePath=args.executable_path,nTauCats=args.ncat,fracCov = args.frac_cov)

    maxGIter = 4
    nChange = 1
    gIter = 0

    while nChange > 0 and gIter < maxGIter:
        assGraph.initNMF()
        print("Round " + str(gIter) + " of gene filtering")
        assGraph.update(200, True,logFile=args.outFileStub + "_log1.txt",drop_strain=None,relax_path=False)

        assGraph.writeGeneError(args.outFileStub + "_" + str(gIter)+ "_geneError.csv")
        
        genesSelect = filterGenes(assGraph)
        nChange = -len(genesSelect) + len(assGraph.genes)
        print("Removed: " + str(nChange) + " genes")
        assemblyGraphsSelect = {s:assemblyGraphs[s] for s in genesSelect}
        source_maps_select = {s:source_maps[s] for s in genesSelect} 
        sink_maps_select = {s:sink_maps[s] for s in genesSelect}

        assGraph = AssemblyPathSVA(prng, assemblyGraphsSelect, source_maps_select, sink_maps_select, G = args.strain_number, readLength=args.readLength,ARD=True,BIAS=True, fgExePath=args.executable_path,nTauCats=args.ncat,fracCov = args.frac_cov)
        
        gIter += 1
    
    assGraph.initNMF()
    
    assGraph.update(300, True,logFile=args.outFileStub + "_log3.txt",drop_strain=None,relax_path=False,uncertainFactor=1.)
  
    #assGraph.update(100, True,logFile=args.outFileStub + "_log3.txt",drop_strain=None,relax_path=True)
  
    assGraph.writeGeneError(args.outFileStub + "geneError.csv")

    assGraph.writeMarginals(args.outFileStub + "margFile.csv")
   
    assGraph.getMaximalUnitigs(args.outFileStub + "Haplo_" + str(assGraph.G),drop_strain=None, relax_path=False)
    
    assGraph.writeMaximals(args.outFileStub + "maxFile.tsv",drop_strain=None)
   
    assGraph.writeGammaMatrix(args.outFileStub + "Gamma.csv") 

    assGraph.writeGammaVarMatrix(args.outFileStub + "varGamma.csv") 
    
    assGraph.writeTheta(args.outFileStub + "Theta.csv") 

    assGraph.writePathDivergence(args.outFileStub + "Diver.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
