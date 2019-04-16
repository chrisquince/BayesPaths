import argparse
import sys
import glob
import numpy as np
import os
import re

from collections import defaultdict
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

def overlapDist(gammaMatrixG, gammaMatrixH):

    dist = 0.
    
    G = gammaMatrixG.shape[0]
    H = gammaMatrixH.shape[0]
    S = gammaMatrixH.shape[1]
    
    GCopy = np.copy(gammaMatrixG)
    
    gSort = np.argsort(np.sum(gammaMatrixG,axis=1))
    hSort =  np.argsort(np.sum(gammaMatrixH,axis=1))
    
    assigned = np.full(H,-1)
    totalOverlap = 0.
    
    for hidx in hSort:
        
        maxOverlap = -1.
        maxG = -1
        
        for gidx in gSort:
            overlap = np.sum(np.minimum(GCopy[gidx,:],gammaMatrixH[hidx,:]))
            
            if overlap > maxOverlap:
                maxOverlap = overlap
                maxG = gidx
        
        totalOverlap += maxOverlap
        assigned[hidx] = maxG
        GCopy[maxG,:] = np.maximum(GCopy[maxG,:] - gammaMatrixH[hidx,:],0)
    
    dSum = max(np.sum(gammaMatrixG),np.sum(gammaMatrixH))
    dist = dSum - totalOverlap
    
    return dist, dist/dSum

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

    import ipdb; ipdb.set_trace()
    
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
    genes = []
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
        genes.append(gene)
    
    #run through individual graphs
    assGraphGenes = {}
    for gene in sorted(genes):
        assGraphGene = AssemblyPathSVA(prng,  {gene:assemblyGraphs[gene]},{gene:source_maps[gene]}, {gene:sink_maps[gene]}, G = args.strain_number, readLength=args.readLength,ARD=True,BIAS=True, fgExePath=args.executable_path,nTauCats=args.ncat,fracCov = args.frac_cov)
   
        assGraphGene.initNMF()   

        assGraphGene.update(200, True,logFile=args.outFileStub + "_log1.txt",drop_strain=None,relax_path=False)
    
        print(gene + "," + str(assGraphGene.calc_elbo()))
            
        assGraphGenes[gene] = assGraphGene

    overlapDists = defaultdict(dict)
    
    for geneI in sorted(genes):
        for geneJ in sorted(genes):
            (distIJ, fDistIJ) = overlapDist(assGraphGenes[geneI].expGamma, assGraphGenes[geneJ].expGamma)
            overlapDists[geneI][geneJ] = fDistIJ
    
    errorDists = defaultdict(dict)
    elboDists = defaultdict(dict)
    
    for geneI in sorted(genes):
        for geneJ in sorted(genes):
            if geneI != geneJ:
                #run geneJ using gamma from geneI
                IG = assGraphGenes[geneI].G
                assGraphGeneJ = AssemblyPathSVA(prng,  {geneJ:assemblyGraphs[geneJ]},{geneJ:source_maps[geneJ]}, {geneJ:sink_maps[geneJ}, G = IG, readLength=args.readLength,ARD=True,BIAS=True, fgExePath=args.executable_path,nTauCats=args.ncat,fracCov = args.frac_cov)
   
                assGraphGeneJ.initNMFGamma(assGraphGenes[geneI].expGamma)   

                assGraphGeneJ.updateGammaFixed(100)


    for geneI in sorted(genes):
        for geneJ in sorted(genes):
            if geneI != geneJ:
                assGraphGeneIJ = AssemblyPathSVA(prng,  {geneI:assemblyGraphs[geneI],geneJ:assemblyGraphs[geneJ]},{geneI:source_maps[geneI],geneJ:source_maps[geneJ]},{geneI:sink_maps[geneI],geneJ:sink_maps[geneJ]}, G = args.strain_number, readLength=args.readLength,ARD=True,BIAS=True, fgExePath=args.executable_path,nTauCats=args.ncat,fracCov = args.frac_cov)
   
                assGraphGeneIJ.initNMF()   

                assGraphGeneIJ.update(200, True,logFile=args.outFileStub + "_log1.txt",drop_strain=None,relax_path=False)
    
                divElbo = assGraphGeneIJ.calc_elbo() - assGraphGenes[geneI].calc_elbo() -  assGraphGenes[geneJ].calc_elbo()
        
                divError = assGraphGeneIJ.mean_diff() - 0.5*(assGraphGenes[geneI].mean_diff() + assGraphGenes[geneJ].mean_diff())
           
                elboDists[geneI][geneJ] = divElbo
                
                errorDists[geneI][geneJ] = divError
    
    



if __name__ == "__main__":
    main(sys.argv[1:])
