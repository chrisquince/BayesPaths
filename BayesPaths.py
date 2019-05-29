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
        if devArray[gidx] > 2.5*medianDevError and error_array[gidx] > medianErr:
            print("Removing: " + str(gene))
        else:
            genesSelect.append(gene)

    return genesSelect 

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("Gene_dir", help="directory with gfa files in")

    parser.add_argument("kmer_length", help="kmer length assumed overlap")

    parser.add_argument("outFileStub", help="output file stub")
    
    parser.add_argument('-l','--cog_list',nargs='?', default=None)

    parser.add_argument('-t','--length_list',nargs='?', default=None, help=("amino acid lengths for genes"))

    parser.add_argument('-f','--frac_cov',nargs='?', default=0.02, type=float, 
        help=("fractional coverage for noise nodes"))

    parser.add_argument('-g','--strain_number',nargs='?', default=5, type=int, 
        help=("maximum number of strains"))

    parser.add_argument('-n','--ncat',nargs='?', default=10, type=int, 
        help=("number of noise categories"))

    parser.add_argument('-r','--readLength',nargs='?', default=100., type=float,
        help=("read length used for sequencing defaults 100bp"))

    parser.add_argument('-s', '--random_seed', default=23724839, type=int,
        help="specifies seed for numpy random number generator defaults to 23724839 applied after random filtering")

    parser.add_argument('-e','--executable_path',nargs='?', default='./runfg_source/', type=str,
        help=("path to factor graph executable"))

    parser.add_argument('-u','--uncertain_factor',nargs='?', default=0.0, type=float,
        help=("penalisation on uncertain strains"))

    parser.add_argument('--relax', dest='relax_path', action='store_true')

    args = parser.parse_args()

    
    np.random.seed(args.random_seed) #set numpy random seed not needed hopefully
    prng = RandomState(args.random_seed) #create prng from seed 

    cogLengths = {}
    if  args.length_list != None:
        with open(args.length_list,'r') as cog_file:
            for line in cog_file:
                line = line.rstrip()
                toks = line.split('\t') 
                cogLengths[toks[0]] = float(toks[1])
        
    
    if args.cog_list == None:
        gfaFiles = glob.glob(args.Gene_dir + '/*.gfa')    
    else:
        with open(args.cog_list,'r') as cog_file:
            cogs = [line.rstrip() for line in cog_file]
        gfaFiles = [args.Gene_dir + "/" + x + ".gfa" for x in cogs]

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
        
        covFile = gfaFile[:-3] + "tsv"
        
        try:
            unitigGraph = UnitigGraph.loadGraphFromGfaFile(gfaFile,int(args.kmer_length), covFile, tsvFile=True, bRemoveSelfLinks = True)
        except IOError:
             print('Trouble using file {}'.format(gfaFile))
             continue
             
        deadEndFile = gfaFile[:-3] + "deadends"
        
        stopFile = gfaFile[:-3] + "stops"
        
        deadEnds = []
        
        try:
            with open(deadEndFile) as f:
                for line in f:
                    line.strip()
                    deadEnds.append(line)
        except IOError:
             print('Trouble using file {}'.format(deadEndFile))
             continue
        
        stops = []
        
        try:
            with open(stopFile) as f:
                for line in f:
                    line = line.strip()
                    toks = line.split("\t")
                    dirn = True
                    if toks[1] == '-':
                        dirn = False
                    stops.append((toks[0],dirn))
        except IOError:
            print('Trouble using file {}'.format(stopFile))
            continue
             
        if gene in cogLengths:
            (source_list, sink_list) = unitigGraph.selectSourceSinksStops(stops, deadEnds, cogLengths[gene]*3)
        else:
            (source_list, sink_list) = unitigGraph.selectSourceSinksStops(stops, deadEnds)
        

        source_names = [convertNodeToName(source) for source in source_list] 
        sink_names = [convertNodeToName(sink) for sink in sink_list]
            
        sink_maps[gene] = sink_list
        source_maps[gene] = source_list
        assemblyGraphs[gene] = unitigGraph
    
    #import ipdb; ipdb.set_trace() 
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
        assGraph.update(500, True,logFile=args.outFileStub + "_log1.txt",drop_strain=None,relax_path=False)

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
    
    assGraph.update(500, True,logFile=args.outFileStub + "_log3.txt",drop_strain=None,relax_path=False,uncertainFactor=args.uncertain_factor)
  
    assGraph.update(500, True,logFile=args.outFileStub + "_log4.txt",drop_strain=None,relax_path=args.relax_path,uncertainFactor=args.uncertain_factor)
  
    assGraph.writeGeneError(args.outFileStub + "geneError.csv")

    assGraph.writeMarginals(args.outFileStub + "margFile.csv")
   
    assGraph.getMaximalUnitigs(args.outFileStub + "Haplo_" + str(assGraph.G),drop_strain=None, relax_path=args.relax_path)
    
    assGraph.writeMaximals(args.outFileStub + "maxFile.tsv",drop_strain=None)
   
    assGraph.writeGammaMatrix(args.outFileStub + "Gamma.csv") 

    assGraph.writeGammaVarMatrix(args.outFileStub + "varGamma.csv") 
    
    assGraph.writeTheta(args.outFileStub + "Theta.csv") 

    assGraph.writeTau(args.outFileStub + "Tau.csv")

    assGraph.writePathDivergence(args.outFileStub + "Diver.csv",relax_path=args.relax_path)


if __name__ == "__main__":
    main(sys.argv[1:])
