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
        if devArray[gidx] > 2.0*medianDevError and error_array[gidx] > medianErr:
            print("Removing: " + str(gene))
        else:
            genesSelect.append(gene)

    return genesSelect 

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("Gene_dir", help="directory with gfa files in")

    parser.add_argument("kmer_length", help="kmer length assumed overlap")

    parser.add_argument("outFileStub", help="output file stub")

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

#    import ipdb; ipdb.set_trace()
    
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
    
    
    assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G = args.strain_number, readLength=args.readLength,ARD=True,BIAS=True, fgExePath=args.executable_path,nTauCats=args.ncat,fracCov = args.frac_cov)
    
    genesRemove = assGraph.get_outlier_cogs_sample(mCogFilter = 3.0, cogSampleFrac=0.80)
    
    genesFilter = list(set(assGraph.genes) ^ set(genesRemove))

    assemblyGraphsFilter = {s:assemblyGraphs[s] for s in genesFilter}
    source_maps_filter = {s:source_maps[s] for s in genesFilter} 
    sink_maps_filter = {s:sink_maps[s] for s in genesFilter}
    
    assGraph = AssemblyPathSVA(prng, assemblyGraphsFilter, source_maps_filter, sink_maps_filter, G = args.strain_number, readLength=args.readLength,ARD=True,BIAS=True, fgExePath=args.executable_path,nTauCats=args.ncat,fracCov = args.frac_cov)

    
    assGraph.initNMF()

    assGraph.update(200, True,logFile=args.outFileStub + "_log1.txt",drop_strain=None,relax_path=False)
 
    genesSelect = filterGenes(assGraph)
 
    assemblyGraphsSelect = {s:assemblyGraphs[s] for s in genesSelect}
    source_maps_select = {s:source_maps[s] for s in genesSelect} 
    sink_maps_select = {s:sink_maps[s] for s in genesSelect}

    assGraphS = AssemblyPathSVA(prng, assemblyGraphsSelect, source_maps_select, sink_maps_select, G = args.strain_number, readLength=args.readLength,ARD=True,BIAS=True, fgExePath=args.executable_path,nTauCats=args.ncat,fracCov = args.frac_cov)
    
    assGraphS.initNMF()

    assGraphS.update(200, True,logFile=args.outFileStub + "_log2.txt",drop_strain=None,relax_path=False)

    genesSelect2 = filterGenes(assGraphS)
 
    assemblyGraphsSelect2 = {s:assemblyGraphs[s] for s in genesSelect2}
    source_maps_select2 = {s:source_maps[s] for s in genesSelect2} 
    sink_maps_select2 = {s:sink_maps[s] for s in genesSelect2}

    assGraphS2 = AssemblyPathSVA(prng, assemblyGraphsSelect2, source_maps_select2, sink_maps_select2, G = args.strain_number, readLength=args.readLength,ARD=True,BIAS=True, fgExePath=args.executable_path,nTauCats=args.ncat,fracCov = args.frac_cov)
    
    assGraphS2.initNMF()
    
    assGraphS2.update(500, True,logFile=args.outFileStub + "_log3.txt",drop_strain=None,relax_path=False)
  
    assGraphS2.update(100, True,logFile=args.outFileStub + "_log3.txt",drop_strain=None,relax_path=True)
  
    gene_mean_error = assGraphS2.gene_mean_diff()
    gene_mean_elbo = assGraphS2.gene_mean_elbo()

    for (gene, error) in gene_mean_error.items():
        print(gene + "," + str(error) + "," + str(gene_mean_elbo[gene]))

    assGraphS2.writeMarginals(args.outFileStub + "margFile.csv")
   
    assGraphS2.getMaximalUnitigs(args.outFileStub + "Haplo_" + str(assGraphS2.G),drop_strain=None, relax_path=True)
 
    assGraphS2.writeMaximals(args.outFileStub + "maxFile.tsv",drop_strain=None)
   
    assGraphS2.writeGammaMatrix(args.outFileStub + "Gamma.csv") 

    assGraphS2.writeGammaVarMatrix(args.outFileStub + "varGamma.csv") 
    
    assGraphS2.writeTheta(args.outFileStub + "Theta.csv") 

if __name__ == "__main__":
    main(sys.argv[1:])
