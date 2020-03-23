import argparse
import sys
import glob
import numpy as np
import os
import re

from Utils.mask import compute_folds_attempts
from collections import defaultdict
from Utils.UnitigGraph import UnitigGraph
from AssemblyPath.AssemblyPathSVAS import AssemblyPathSVA
from Utils.UtilsFunctions import convertNodeToName
from numpy.random import RandomState
from pathos.multiprocessing import ProcessingPool 
COG_COV_DEV = 2.5


def filterGenes(assGraph, bGeneDev):
    gene_mean_error = assGraph.gene_mean_diff()
    gene_mean_elbo = assGraph.gene_mean_elbo()
    gene_mean_dev = assGraph.gene_mean_deviance()
    
    if bGeneDev:
        eval_error = gene_mean_dev
    else:
        eval_error = gene_mean_error
    
    errors = []
    genes = []
    
    for (gene, error) in eval_error.items():
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

def selectSamples(assGraph, genesSelect, readLength,kLength,minCov,minFracCov):
        
    nGenes = len(genesSelect)
        
    g = 0
    geneSampleCovArray = np.zeros((nGenes,assGraph.S))
    for gene in genesSelect:
        geneSampleCovArray[g,:] = assGraph.geneCovs[gene]
        g = g + 1    
    
    kFactor = readLength/(readLength - kLength + 1.)
    
    sampleMean = np.mean(geneSampleCovArray,axis=0)*kFactor
        
    minCov = max(minCov,minFracCov*np.max(sampleMean))
    
    return sampleMean > minCov


def assGraphWorker(gargs):

    (prng, assemblyGraphs, source_maps, sink_maps, G, r, args, selectedSamples, outDir, M_train, M_test) = gargs 

    assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G, args.readLength,
                                ARD=False,BIAS=args.bias,  NOISE=args.NOISE, fgExePath=args.executable_path, bLoess = args.loess, 
                                bGam = args.usegam, tauType = args.tauType, 
                                fracCov = args.frac_cov, noiseFrac = args.noise_frac)
    
    assGraph.initNMF(M_train)
                
    assGraph.update(args.iters, False, M_train, True, args.outFileStub + "_log4.txt",drop_strain=None,relax_path=False, bMulti = False)

    assGraph.update(args.iters, False, M_train, True, args.outFileStub + "_log4.txt",drop_strain=None,relax_path=False, bMulti = False)

    assGraph.updateTau(False, M_test, True)

    assGraph.writeOutput(outDir + "/Run" + '_g' + str(G) + "_r" + str(r), False, selectedSamples, M_test)

    train_elbo = assGraph.calc_elbo(M_test, True)
    train_err  = assGraph.predict_sqrt(M_test,True)
    train_errP = assGraph.predictMaximal(M_test, True)
    train_div = assGraph.div(M_test, True)
    train_divF = assGraph.divF(M_test, True)
    train_ll = assGraph.calc_expll_poisson(M_test, True)
    
    fitFile = outDir + "/Run" + '_g' + str(G) + "_r" + str(r) + "_fit.txt"
    with open(fitFile, 'w') as ffile:
        ffile.write(str(train_elbo) + "," + str(train_err)  + "," + str(train_errP) + "," + str(train_div) + "," + str(train_divF) + "," + str(train_ll) + "\n")

    return (train_elbo, train_err, train_errP, train_div, train_divF, train_ll, assGraph.G)

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

    parser.add_argument('-m','--min_cov',nargs='?', default=1.0, type=float,
        help=("min. sample coverage"))

    parser.add_argument('-mf','--min_frac_cov',nargs='?', default=0.05, type=float,
        help=("min. fractional sample coverage"))

    parser.add_argument('-g','--strain_number',nargs='?', default=5, type=int, 
        help=("maximum number of strains"))

    parser.add_argument('-p','--paths_file',nargs='?', default=None)
        
    parser.add_argument('--loess', dest='loess', action='store_true')
    
    parser.add_argument('--no_gam', dest='usegam', action='store_false')

    parser.add_argument('--no_ard', dest='ARD', action='store_false')

    parser.add_argument('--no_noise', dest='NOISE', action='store_false')

    parser.add_argument('-i', '--iters', default=250, type=int,
        help="number of iterations for the variational inference")
    
    parser.add_argument('-nfo', '--nofolds', default=10, type=int,
        help="number of folds for the CV analysis")

    parser.add_argument('-r','--readLength',nargs='?', default=100., type=float,
        help=("read length used for sequencing defaults 100bp"))

    parser.add_argument('-s', '--random_seed', default=23724839, type=int,
        help="specifies seed for numpy random number generator defaults to 23724839 applied after random filtering")

    parser.add_argument('-e','--executable_path',nargs='?', default='./runfg_source/', type=str,
        help=("path to factor graph executable"))

    parser.add_argument('-u','--uncertain_factor',nargs='?', default=5., type=float,
        help=("penalisation on uncertain strains"))

    parser.add_argument('--nofilter', dest='filter', action='store_false')

    parser.add_argument('--no_run_elbow', dest='run_elbow', action='store_false')

    parser.add_argument('--norelax', dest='relax_path', action='store_false')

    parser.add_argument('--nobias', dest='bias', action='store_false')

    parser.add_argument('--tau_type', dest='tauType', default='auto',choices=['fixed','log','empirical','auto','poisson'],help='Strategy for setting precision')

    parser.add_argument('--nogenedev', dest='bGeneDev', action='store_false')

    args = parser.parse_args()

    #import ipdb; ipdb.set_trace()    
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
        
        if len(sink_list) > 0 and len(source_list) > 0:        
            sink_maps[gene] = sink_list
            source_maps[gene] = source_list
            assemblyGraphs[gene] = unitigGraph
    
    #import ipdb; ipdb.set_trace() 
    
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
                                ARD=True,BIAS=args.bias, NOISE=args.NOISE, fgExePath=args.executable_path, bLoess = args.loess, bGam = args.usegam, tauType = args.tauType, 
                                fracCov = args.frac_cov, noiseFrac = args.noise_frac)
        
        
        for g in strains:
            for gene, factorGraph in assGraph.factorGraphs.items():
                unitigs = assGraph.assemblyGraphs[gene].unitigs
                
                assGraph.updateExpPhi(unitigs,assGraph.mapGeneIdx[gene],margPhiFixed[g][gene],g)
        
        
    
    assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G = args.strain_number, readLength=args.readLength,
                                ARD=True,BIAS=args.bias,  NOISE=args.NOISE, fgExePath=args.executable_path, bLoess = args.loess, 
                                bGam = args.usegam, tauType = args.tauType,
                                fracCov = args.frac_cov, noiseFrac = args.noise_frac)
    
    genesRemove = assGraph.get_outlier_cogs_sample(mCogFilter = 3.0, cogSampleFrac=0.80)
    
    genesWidth = assGraph.calcTreeWidths()
    
    for (gene,width) in genesWidth.items():
        if width > 16 or width < 0:
            if gene not in genesRemove:
                genesRemove.append(gene)

    genesFilter = list(set(assGraph.genes) ^ set(genesRemove))

    selectedSamples = selectSamples(assGraph, genesFilter, float(args.readLength),float(args.kmer_length),float(args.min_cov),float(args.min_frac_cov))
    
    print('Selecting ' + str(np.sum(selectedSamples)) + ' samples:')
        
    if np.sum(selectedSamples) < 2:
        print("Not recommended to use bias with fewer than 2 samples setting bias to false and using fixed tau")
        args.bias = False
        args.tauType = 'fixed'

    if np.sum(selectedSamples) < 1:
        summaryFile=args.outFileStub + "_summary.txt"
        with open(summaryFile,'w') as f:
            print("Not recommended to run BayesPaths.py with fewer than 1 samples exiting...")
            f.write('Not recommended to run BayesPaths.py with fewer than 1 samples exiting..\n')
        sys.exit(0)

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
                                ARD=True,BIAS=args.bias,  NOISE=args.NOISE, fgExePath=args.executable_path, bLoess = args.loess, bGam = args.usegam, 
                                tauType = args.tauType, fracCov = args.frac_cov,  noiseFrac = args.noise_frac)


    assemblyGraphs = assemblyGraphsFilter
    source_maps = source_maps_filter
    sink_maps = sink_maps_filter

    if args.filter:
        maxGIter = 4
        nChange = 1
        gIter = 0

        while nChange > 0 and gIter < maxGIter:
            assGraph.initNMF()
            print("Round " + str(gIter) + " of gene filtering")
            assGraph.update(args.iters*2, True, None, True, logFile=args.outFileStub + "_log1.txt",drop_strain=None,relax_path=False)
            #MSEP = assGraph.predictMaximal(np.ones((assGraph.V,assGraph.S)))
            #MSE = assGraph.predict(np.ones((assGraph.V,assGraph.S)))
            assGraph.writeGeneError(args.outFileStub + "_" + str(gIter)+ "_geneError.csv")
        
            assGraph.writeOutput(args.outFileStub + '_G' + str(gIter), False, selectedSamples)

            genesSelect = filterGenes(assGraph,args.bGeneDev)
            nChange = -len(genesSelect) + len(assGraph.genes)
            print("Removed: " + str(nChange) + " genes")
            assemblyGraphsSelect = {s:assemblyGraphs[s] for s in genesSelect}
            source_maps_select = {s:source_maps[s] for s in genesSelect} 
            sink_maps_select = {s:sink_maps[s] for s in genesSelect}

            assGraph = AssemblyPathSVA(prng, assemblyGraphsSelect, source_maps_select, sink_maps_select, G = args.strain_number, readLength=args.readLength,
                                        ARD=True,BIAS=args.bias,  NOISE=args.NOISE, fgExePath=args.executable_path, bLoess = args.loess, bGam = args.usegam, 
                                        tauType = args.tauType, fracCov = args.frac_cov, noiseFrac = args.noise_frac)


            assemblyGraphs = assemblyGraphsSelect
            source_maps = source_maps_select
            sink_maps = sink_maps_select

            gIter += 1
    

    assGraph.initNMF()

    assGraph.update(args.iters, True, None, True, logFile=args.outFileStub + "_log2.txt",drop_strain=None,relax_path=False,bMulti=True)

    assGraph.update(args.iters, True, None, True, logFile=args.outFileStub + "_log2.txt",drop_strain=None,relax_path=args.relax_path)

    assGraph.writeOutput(args.outFileStub, False, selectedSamples)

    assGraph.update(args.iters, True, None, True, logFile=args.outFileStub + "_log3.txt",drop_strain=None,relax_path=False,uncertainFactor=args.uncertain_factor)

    assGraph.update(args.iters, True, None, True, logFile=args.outFileStub + "_log3.txt",drop_strain=None,relax_path=args.relax_path)
  
    assGraph.writeOutput(args.outFileStub + "_P", False, selectedSamples)

    Gopt = assGraph.G

    if (args.run_elbow and Gopt >= 4) and assGraph.S >=5:
        no_folds=int(args.nofolds)
    
        elbos = defaultdict(lambda: np.zeros(no_folds))
        errs = defaultdict(lambda: np.zeros(no_folds))
        errsP = defaultdict(lambda: np.zeros(no_folds))
        divs = defaultdict(lambda: np.zeros(no_folds))
        divFs = defaultdict(lambda: np.zeros(no_folds))
        expLLs = defaultdict(lambda: np.zeros(no_folds))
        Hs = defaultdict(lambda: np.zeros(no_folds))    
        
        
        M_attempts = 1000

        M = np.ones((assGraph.V,assGraph.S))

        (Ms_train,Ms_test) = compute_folds_attempts(I=assGraph.V,J=assGraph.S,no_folds=no_folds,attempts=M_attempts,M=M)


        outDir = os.path.dirname(args.outFileStub  + "/CVAnalysis")
        try:
            os.mkdir(outDir)

        except FileExistsError:
            print('Directory not created.')


        for g in range(1,Gopt + 1):
                        
            fold_p = ProcessingPool(processes=no_folds)
            
            results = []
            pargs = []
            for f in range(no_folds):
                
                M_train = Ms_train[f]
                M_test = Ms_test[f]
                
                prng = RandomState(args.random_seed + f) 
                
                pargs.append([prng, assemblyGraphs, source_maps, sink_maps, g, f, args, selectedSamples, outDir, M_train, M_test])            

            results = fold_p.amap(assGraphWorker,pargs)

            resultsa = list(results.get())
            
            for f in range(no_folds):
                elbos[g][f] = resultsa[f][0]
                errs[g][f]  =  resultsa[f][1]
                errsP[g][f] = resultsa[f][2]
                divs[g][f]  = resultsa[f][3]
                divFs[g][f] = resultsa[f][4]
                expLLs[g][f] = resultsa[f][5]
                Hs[g][f] = resultsa[f][6]
                
        mean_errs = np.zeros(Gopt)
        median_hs = np.zeros(Gopt)
        with open(args.outFileStub + "_CV.csv",'w') as f:
            f.write("No_strains,mean_elbo,mean_err,mean_div,mean_divF,median_h\n")
            for g in range(1,Gopt + 1):
                mean_elbo = np.mean(elbos[g])        
                mean_err = np.mean(errs[g])
                
                mean_errP = np.mean(errsP[g])   
                mean_div = np.mean(divs[g]) 
                mean_divF = np.mean(divFs[g])     
                median_h = np.median(Hs[g])
                median_hs[g - 1] = median_h 
                median_ll = np.mean(expLLs[g]) 
                
                mean_errs[g - 1] = median_ll
                f.write(str(g) +"," + str(mean_elbo) +"," + str(mean_err) + "," + str(mean_errP) + "," + str(mean_div) + "," + str(mean_divF) + "," + str(median_ll) + "," + str(median_h) + '\n')
                print(str(g) +"," + str(mean_elbo) +"," + str(mean_err) + "," + str(mean_errP) + "," + str(mean_div) + "," + str(mean_divF) + "," + str(median_ll) + "," + str(median_h))
    
        #Rerun with optimal g
    
        minG = int(median_hs[np.argmax(mean_errs)]) 
       
        print("Using " + str(minG) + " strains")
 
        assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G = minG, readLength=args.readLength,
                                        ARD=True,BIAS=args.bias,  NOISE=args.NOISE, fgExePath=args.executable_path, bLoess = args.loess, bGam = args.usegam, 
                                        tauType = args.tauType, fracCov = args.frac_cov, noiseFrac = args.noise_frac)
   
        assGraph.initNMF()

        assGraph.update(args.iters, True, None, True, logFile=args.outFileStub + "_log6.txt",drop_strain=None,relax_path=False,bMulti=True)    

        assGraph.update(args.iters, True, None, True, logFile=args.outFileStub + "_log6.txt",drop_strain=None,relax_path=False,bMulti=True)

        assGraph.update(args.iters, True, None, True, logFile=args.outFileStub + "_log6.txt",drop_strain=None,relax_path=args.relax_path)

    
    assGraph.writeOutput(args.outFileStub + "_Q", False, selectedSamples)
    
    
    summaryFile=args.outFileStub + "_summary.txt"
    with open(summaryFile,'w') as f:
        print("BayesPaths finished")
        f.write("BayesPaths finished resolving " + str(Gopt) + " strains")
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])
