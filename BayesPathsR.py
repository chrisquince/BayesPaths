import argparse
import sys
import glob
import numpy as np
import os
import re

from collections import defaultdict
from GraphProcess import getMaximumCoverageWalk
from Utils.UnitigGraph import UnitigGraph
from AssemblyPath.AssemblyPathSVAR import AssemblyPathSVA
from Utils.UtilsFunctions import convertNodeToName
from numpy.random import RandomState

COG_COV_DEV = 2.5
SAMPLE_MIN_COV = 1.0

def filterGenes(assGraph):
    gene_mean_error = assGraph.gene_mean_diff()
    gene_mean_elbo = assGraph.gene_mean_elbo()
    gene_mean_dev = assGraph.gene_mean_deviance()
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
        if devArray[gidx] > COG_COV_DEV*medianDevError and error_array[gidx] > medianErr:
            print("Removing: " + str(gene))
        else:
            genesSelect.append(gene)

    return genesSelect 

def selectSamples(assGraph, genesSelect):
        
    nGenes = len(genesSelect)
        
    g = 0
    geneSampleCovArray = np.zeros((nGenes,assGraph.S))
    for gene in genesSelect:
        geneSampleCovArray[g,:] = assGraph.geneCovs[gene]
        g = g + 1    
    
    sampleMean = np.mean(geneSampleCovArray,axis=0)
        
    minCov = max(SAMPLE_MIN_COV,0.05*np.max(sampleMean))

    return sampleMean > minCov


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("Gene_dir", help="directory with gfa files in")

    parser.add_argument("kmer_length", help="kmer length assumed overlap")

    parser.add_argument("outFileStub", help="output file stub")
    
    parser.add_argument('-l','--cog_list',nargs='?', default=None)

    parser.add_argument('-t','--length_list',nargs='?', default=None, help=("amino acid lengths for genes"))

    parser.add_argument('-f','--frac_cov',nargs='?', default=0.02, type=float, 
        help=("fractional coverage for noise nodes"))

    parser.add_argument('-nf','--noise_frac',nargs='?', default=0.02, type=float,
        help=("fractional coverage for noise category"))

    parser.add_argument('-g','--strain_number',nargs='?', default=5, type=int, 
        help=("maximum number of strains"))

    parser.add_argument('-n','--ncat',nargs='?', default=10, type=int, 
        help=("number of noise categories"))

    parser.add_argument('-p','--paths_file',nargs='?', default=None)

    parser.add_argument('-gf','--gamma_file',nargs='?', default=None)

    parser.add_argument('-y','--tautype',nargs='?', default='Adaptive', type=str,
        help=("type of variance model"))
        
    parser.add_argument('--noreassign', dest='reassign', action='store_false')

    parser.add_argument('--nocogs', dest='cogs', action='store_false')
    
    parser.add_argument('-r','--readLength',nargs='?', default=100., type=float,
        help=("read length used for sequencing defaults 100bp"))

    parser.add_argument('-s', '--random_seed', default=23724839, type=int,
        help="specifies seed for numpy random number generator defaults to 23724839 applied after random filtering")

    parser.add_argument('-e','--executable_path',nargs='?', default='./runfg_source/', type=str,
        help=("path to factor graph executable"))

    parser.add_argument('-u','--uncertain_factor',nargs='?', default=0.1, type=float,
        help=("penalisation on uncertain strains"))

    parser.add_argument('--norelax', dest='relax_path', action='store_false')

    parser.add_argument('--nobias', dest='bias', action='store_false')

    args = parser.parse_args()

    import ipdb; ipdb.set_trace()    
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

        if args.cogs:
            p = re.compile('COG[0-9]+')

            m = p.search(gfaFile)
        
            if m is None:
                raise ValueError

            gene = m.group()
        else:
            base=os.path.basename(gfaFile)
            gene = base[:-4]

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
            (source_list, sink_list) = unitigGraph.selectSourceSinksStops(stops, deadEnds,400)
        

        source_names = [convertNodeToName(source) for source in source_list] 
        sink_names = [convertNodeToName(sink) for sink in sink_list]
        
        if len(sink_list) > 0 and len(source_list) > 0:        
            sink_maps[gene] = sink_list
            source_maps[gene] = source_list
            assemblyGraphs[gene] = unitigGraph
    
    #import ipdb; ipdb.set_trace() 
    
    if args.gamma_file != None:
        
        with open(args.gamma_file,'r') as g:
            gidx = 0
            mapG = {}
        
            header = next(g)

            toks = header.split(',')

            toks.pop(0)

            selectIndices = np.asarray([int(x) for x in toks])


            for line in g:
                line = line.rstrip()
                
                toks = line.split(',')
            
                mapG[int(gidx)] = np.asarray([float(x) for x in toks[1:]])
            
                gidx += 1
            
            S = mapG[0].shape[0]
            G = gidx
            gammaFixed = np.zeros((G,S))
            for gidx in range(G):
                gammaFixed[gidx,:] = mapG[gidx]

            graphDash = next(iter(assemblyGraphs.values()))
            SDash = next(iter(graphDash.covMap.values())).shape[0]

            selectedSamples = np.zeros(SDash,dtype=np.bool)
            selectedSamples[selectIndices] = True             

            for gene, graph in assemblyGraphs.items():
                graph.selectSamples(selectedSamples)            
            
            assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G, readLength=args.readLength,
                                ARD=True,BIAS=args.bias, NOISE=False, fgExePath=args.executable_path, tauType = args.tautype, nTauCats=args.ncat,bReassign=args.reassign,
                                fracCov = args.frac_cov, noiseFrac = args.noise_frac)
            
            assGraph.initNMFGammaFixed(gammaFixed)
            
            assGraph.updateGammaFixed(25,relax_path=True)

            assGraph.writeGeneError(args.outFileStub + "geneError.csv")

            assGraph.writeMarginals(args.outFileStub + "margFile.csv")
   
            assGraph.getMaximalUnitigs(args.outFileStub + "Haplo_" + str(assGraph.G),drop_strain=None, relax_path=True)
    
            assGraph.writeMaximals(args.outFileStub + "maxFile.tsv",drop_strain=None)
   
            if assGraph.BIAS:
                assGraph.writeTheta(args.outFileStub + "Theta.csv") 

            assGraph.writeTau(args.outFileStub + "Tau.csv")

            #assGraph.writePathDivergence(args.outFileStub + "Diver.csv",relax_path=relax_path_out)

            assGraph.writePredictions(args.outFileStub + "Pred.csv" , drop_strain=None)

            sys.exit(0)
    
    if  args.paths_file != None:
    
        margPhiFixed = defaultdict(lambda: defaultdict(dict))
        strains = set()
        cogs = set()
        
        with open(args.paths_file,'r') as paths_file:

            for line in paths_file:
                line = line.rstrip()
                
                if line.startswith('>'):
                    toks = line[1:].split('_')
                    
                    g = int(toks[1])
                    strains.add(g)
                    
                    cog = toks[0]
                    cogs.add(cog)
                    
                    line = paths_file.readline()
                    
                    line = line.rstrip()
                    
                    toks = line.split(',')
                    
                    for tok in toks:
                        margPhiFixed[g][cog][tok[:-1]] = np.asarray([0.0,1.0])
                
        G = len(strains)
        genesFilter = list(cogs)
        assemblyGraphsFilter = {s:assemblyGraphs[s] for s in genesFilter}
        source_maps_filter = {s:source_maps[s] for s in genesFilter} 
        sink_maps_filter = {s:sink_maps[s] for s in genesFilter}
    
        assGraph = AssemblyPathSVA(prng, assemblyGraphsFilter, source_maps_filter, sink_maps_filter, G, readLength=args.readLength,
                                ARD=True,BIAS=args.bias, fgExePath=args.executable_path, tauType = args.tautype, nTauCats=args.ncat,bReassign=args.reassign,
                                fracCov = args.frac_cov, noiseFrac = args.noise_frac)
        
        
        for g in strains:
            for gene, factorGraph in assGraph.factorGraphs.items():
                unitigs = assGraph.assemblyGraphs[gene].unitigs
                
                assGraph.updateExpPhi(unitigs,assGraph.mapGeneIdx[gene],margPhiFixed[g][gene],g)
        
        
    
    assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G = args.strain_number, readLength=args.readLength,
                                ARD=True,BIAS=args.bias, fgExePath=args.executable_path,tauType = args.tautype, nTauCats=args.ncat,bReassign=args.reassign,
                                fracCov = args.frac_cov, noiseFrac = args.noise_frac)
    
    genesRemove = assGraph.get_outlier_cogs_sample(mCogFilter = 3.0, cogSampleFrac=0.80)
    
    genesFilter = list(set(assGraph.genes) ^ set(genesRemove))

    selectedSamples = selectSamples(assGraph, genesFilter)
    
    if  np.sum(selectedSamples) < assGraph.S:
        print('Selecting samples:')
        
        selectedIndices = np.where(selectedSamples)
    
        sString = ','.join([str(s) for s in selectedIndices])
    
        print(sString)
        logFile=args.outFileStub + "_log1.txt"
        with open(logFile,'w') as f:
            f.write(sString + '\n')
    
    assemblyGraphsFilter = {s:assemblyGraphs[s] for s in genesFilter}
    source_maps_filter = {s:source_maps[s] for s in genesFilter} 
    sink_maps_filter = {s:sink_maps[s] for s in genesFilter}
    
    for gene, graph in assemblyGraphsFilter.items():
        graph.selectSamples(selectedSamples)
    
    assGraph = AssemblyPathSVA(prng, assemblyGraphsFilter, source_maps_filter, sink_maps_filter, G = args.strain_number, readLength=args.readLength,
                                ARD=True,BIAS=args.bias, fgExePath=args.executable_path, tauType = args.tautype, nTauCats=args.ncat,bReassign=args.reassign,
                                fracCov = args.frac_cov,  noiseFrac = args.noise_frac)

    maxGIter = 4
    nChange = 1
    gIter = 0

    while nChange > 0 and gIter < maxGIter:
        assGraph.initNMF()
        print("Round " + str(gIter) + " of gene filtering")
        assGraph.update(500, True,logFile=args.outFileStub + "_log1.txt",drop_strain=None,relax_path=False)

        assGraph.writeGeneError(args.outFileStub + "_" + str(gIter)+ "_geneError.csv")
        
        assGraph.writeOutput(args.outFileStub + '_G' + str(gIter), False)

        genesSelect = filterGenes(assGraph)
        nChange = -len(genesSelect) + len(assGraph.genes)
        print("Removed: " + str(nChange) + " genes")
        assemblyGraphsSelect = {s:assemblyGraphs[s] for s in genesSelect}
        source_maps_select = {s:source_maps[s] for s in genesSelect} 
        sink_maps_select = {s:sink_maps[s] for s in genesSelect}

        assGraph = AssemblyPathSVA(prng, assemblyGraphsSelect, source_maps_select, sink_maps_select, G = args.strain_number, readLength=args.readLength,
                                    ARD=True,BIAS=args.bias, fgExePath=args.executable_path,tauType = args.tautype, nTauCats=args.ncat,bReassign=args.reassign,
                                    fracCov = args.frac_cov, noiseFrac = args.noise_frac)
        
        gIter += 1
    

    assGraph.initNMF()

    assGraph.update(250, True,logFile=args.outFileStub + "_log2.txt",drop_strain=None,relax_path=False)

    assGraph.update(250, True,logFile=args.outFileStub + "_log2.txt",drop_strain=None,relax_path=args.relax_path)

    assGraph.writeOutput(args.outFileStub, False)

    assGraph.update(250, True,logFile=args.outFileStub + "_log3.txt",drop_strain=None,relax_path=False,uncertainFactor=args.uncertain_factor)

    assGraph.update(250, True,logFile=args.outFileStub + "_log3.txt",drop_strain=None,relax_path=args.relax_path)
  
    assGraph.writeOutput(args.outFileStub + "_P", False)

if __name__ == "__main__":
    main(sys.argv[1:])
