import argparse
import sys
import glob
import numpy as np
import os
import re

from GraphProcess import getMaximumCoverageWalk
from Utils.UnitigGraph import UnitigGraph
from AssemblyPath.AssemblyPathSVAK import AssemblyPathSVA
from Utils.UtilsFunctions import convertNodeToName
from numpy.random import RandomState

def filterGenes(assGraph,M_test):
    gene_mean_error = assGraph.gene_mean_diff(M_test)
    gene_mean_elbo = assGraph.gene_mean_elbo(M_test)

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

    parser.add_argument('--relax', dest='relax_path', action='store_true')

    args = parser.parse_args()

    #import ipdb; ipdb.set_trace()
    
    np.random.seed(args.random_seed) #set numpy random seed not needed hopefully
    prng = RandomState(args.random_seed) #create prng from seed 
    
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
        
        unitigGraph = UnitigGraph.loadGraphFromGfaFile(gfaFile,int(args.kmer_length), covFile, tsvFile=True)
            
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

    IdentityM = np.ones((assGraph.V,assGraph.S))

    while nChange > 0 and gIter < maxGIter:
        assGraph.initNMF()
        print("Round " + str(gIter) + " of gene filtering")
        assGraph.update(60, True,IdentityM,logFile=args.outFileStub + "_log1.txt",drop_strain=None,relax_path=False)

        assGraph.writeGeneError(args.outFileStub + "_" + str(gIter)+ "_geneError.csv")
        
        genesSelect = filterGenes(assGraph,IdentityM)
        nChange = -len(genesSelect) + len(assGraph.genes)
        print("Removed: " + str(nChange) + " genes")
        assemblyGraphsSelect = {s:assemblyGraphs[s] for s in genesSelect}
        source_maps_select = {s:source_maps[s] for s in genesSelect} 
        sink_maps_select = {s:sink_maps[s] for s in genesSelect}

        assGraph = AssemblyPathSVA(prng, assemblyGraphsSelect, source_maps_select, sink_maps_select, G = args.strain_number, readLength=args.readLength,ARD=True,BIAS=True, fgExePath=args.executable_path,nTauCats=args.ncat,fracCov = args.frac_cov)
        
        gIter += 1
    
    
    ''' Generate matrices M - one list of M's for each value of K. '''
    values_K = range(self.G)
    M_attempts = 1000
    no_folds = 10
    M = np.ones((assGraph.V,assGraph.S))
    all_Ms_training_and_test = [
        compute_folds_attempts(I=assGraph.V,J=assGraph.S,no_folds=no_folds,attempts=M_attempts,M=M)
        for K in values_K
    ]
    
    all_performances = {} 
    average_performances = {}
    
    for K,(Ms_train,Ms_test) in zip(values_K,all_Ms_training_and_test):
    
        assGraph = AssemblyPathSVA(prng, assemblyGraphsSelect, source_maps_select, sink_maps_select, G = K, readLength=args.readLength,ARD=True,BIAS=True, fgExePath=args.executable_path,nTauCats=args.ncat,fracCov = args.frac_cov)
       
        performances = []
        for fold,(M_train,M_test) in enumerate(zip(Ms_train,Ms_test)):
            print "Fold %s of K=%s." % (fold+1, K)
    
            assGraph.initNMF(M_train)

            assGraph.update(200, True, M_train,logFile=None,drop_strain=None,relax_path=True)
           
            train_elbo = assGraph.calc_elbo(M_test)
            train_err  = assGraph.predict(M_test)
            
            print(str(K) + "," + str(assGraph.G) + "," + str(fold) +","+str(train_elbo) + "," + str(train_err))
            performances.append(train_err) 
            fold += 1
        average_performances[K] = sum(performances)/no_folds
    
    for k in range(K):
        print(str(k) + ',' + str(average_performances[k]))


if __name__ == "__main__":
    main(sys.argv[1:])
