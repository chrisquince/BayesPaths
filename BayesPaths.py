import argparse
import sys
import glob
import numpy as np
import os
import re
import logging

from Utils.mask import compute_folds_attempts
from collections import defaultdict
from Utils.UnitigGraph import UnitigGraph
from AssemblyPath.AssemblyPathSVAV import AssemblyPathSVA
from Utils.UtilsFunctions import convertNodeToName
from numpy.random import RandomState
from pathos.multiprocessing import ProcessingPool 
COG_COV_DEV = 2.5


def filterGenes(assGraph, bGeneDev):
    gene_mean_error = assGraph.gene_mean_diff()
    gene_mean_elbo = assGraph.gene_mean_elbo()
    gene_mean_dev = assGraph.gene_mean_poisson()
    
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

    assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G, readLength=args.readLength,
                                ARD=False,BIAS=args.bias,  NOISE=args.NOISE, fgExePath=args.executable_path, bLoess = args.loess, 
                                bGam = args.usegam, tauType = args.tauType, biasType = args.biasType,
                                fracCov = args.frac_cov, noiseFrac = args.noise_frac)
    
    assGraph.initNMFVB2(M_train, True)
                
    #assGraph.writeOutput(outDir + "/RunN" + '_g' + str(G) + "_r" + str(r), False, selectedSamples, M_test)
                
    assGraph.update(args.iters, False, M_train, True, args.outFileStub + "_log4.txt",drop_strain=None,relax_path=False, bMulti = False)

    assGraph.update(args.iters, False, M_train, True, args.outFileStub + "_log4.txt",drop_strain=None,relax_path=False, bMulti = False)

    assGraph.updateTau(False, M_test, True)

    assGraph.writeOutput(outDir + "/Run" + '_g' + str(G) + "_r" + str(r), False, selectedSamples, M_test)

    train_elbo = assGraph.calc_elbo(M_test, True)
    train_err  = assGraph.predict_sqrt(M_test,True)
    train_ll   = assGraph.calc_expll_poisson(M_test, True, False)
    train_ll2  = assGraph.calc_expll_poisson_maximal(M_test, True)
    total_ll2  = assGraph.calc_expll_poisson_maximal(mask = None, bMaskDegen = True)
    
    fitFile = outDir + "/Run" + '_g' + str(G) + "_r" + str(r) + "_fit.txt"
    with open(fitFile, 'w') as ffile:
        ffile.write(str(train_elbo) + "," + str(train_err)  + "," + str(train_ll) + "," + str(train_ll2) + "," + str(total_ll2) + "\n")

    return (train_elbo, train_err, train_ll, train_ll2, total_ll2, assGraph.G, 
                assGraph.expGamma,assGraph.expGamma2, assGraph.expPhi, assGraph.expPhi2, assGraph.expTheta, assGraph.expTheta2)

def readCogLengthFile(cogFile):
    
    cogLengths = {}        

    with open(cogFile,'r') as cog_file:
        for line in cog_file:
            line = line.rstrip()
            toks = line.split('\t') 
            cogLengths[toks[0]] = float(toks[1])

    return cogLengths
    
def readGFA(gfaFiles, kMerLength, cogLengths):

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
            logging.info('Reading %s gfa file', gfaFile)
            unitigGraph = UnitigGraph.loadGraphFromGfaFile(gfaFile,kMerLength, covFile, tsvFile=True, bRemoveSelfLinks = True)
        except IOError:
            logging.ERROR('Trouble using file {}'.format(gfaFile))
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
            logging.ERROR('Trouble using file {}'.format(deadEndFile))
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
            logging.ERROR('Trouble using file {}'.format(stopFile))
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
            

    return (assemblyGraphs,  sink_maps, source_maps, cov_maps)

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

    parser.add_argument('-u','--uncertain_factor',nargs='?', default=2.0, type=float,
        help=("penalisation on uncertain strains"))

    parser.add_argument('--nofilter', dest='filter', action='store_false')

    parser.add_argument('--no_run_elbow', dest='run_elbow', action='store_false')

    parser.add_argument('--norelax', dest='relax_path', action='store_false')

    parser.add_argument('--nobias', dest='bias', action='store_false')
    
    parser.add_argument('--bias_type', dest='biasType', default='unitig',choices=['unitig','gene','bubble'],help='Strategy for setting coverage bias')

    parser.add_argument('--tau_type', dest='tauType', default='auto',choices=['fixed','log','empirical','auto','poisson'],help='Strategy for setting precision')

    parser.add_argument('--nogenedev', dest='bGeneDev', action='store_false')

    args = parser.parse_args()
    
    np.random.seed(args.random_seed) #set numpy random seed not needed hopefully
    
    np.seterr(divide='ignore', invalid='ignore') #tired of all those warnings :(
    
    prng = RandomState(args.random_seed) #create prng from seed 

    #set log file
    logFile=args.outFileStub + "_log.txt"
    logging.basicConfig(format='%(asctime)s %(message)s',filename=logFile, filemode='w', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    cogLengths = {}
    if  args.length_list != None:
        logging.info('Reading from cog length file %s', args.length_list)
        cogLengths = readCogLengthFile(args.length_list)
        logging.info('Read %d cog lengths', len(cogLengths))
    
    if args.cog_list == None:
        gfaFiles = glob.glob(args.Gene_dir + '/*.gfa')
        logging.info('Processing %d .gfa files from directory %s', len(gfaFiles),args.Gene_dir)
    else:
        with open(args.cog_list,'r') as cog_file:
            cogs = [line.rstrip() for line in cog_file]
        gfaFiles = [args.Gene_dir + "/" + x + ".gfa" for x in cogs]
        logging.info('Processing %d .gfa files from file %s', len(gfaFiles),args.cog_list)
    
    
    (assemblyGraphs,  sink_maps, source_maps, cov_maps) = readGFA(gfaFiles,int(args.kmer_length),cogLengths)
    
    #import ipdb; ipdb.set_trace() 
    
    assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G = args.strain_number, 
                                readLength=args.readLength, ARD=True,BIAS=args.bias,  NOISE=args.NOISE, 
                                fgExePath=args.executable_path, bLoess = args.loess, 
                                bGam = args.usegam, tauType = args.tauType, biasType = args.biasType,
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
    #import ipdb; ipdb.set_trace()
    assGraph = AssemblyPathSVA(prng, assemblyGraphsFilter, source_maps_filter, sink_maps_filter, G = args.strain_number, readLength=args.readLength,
                                ARD=True,BIAS=args.bias,  NOISE=args.NOISE, fgExePath=args.executable_path, bLoess = args.loess, bGam = args.usegam, 
                                tauType = args.tauType, biasType = args. biasType, fracCov = args.frac_cov,  noiseFrac = args.noise_frac)


    assemblyGraphs = assemblyGraphsFilter
    source_maps = source_maps_filter
    sink_maps = sink_maps_filter

    if args.filter:
        maxGIter = 4
        nChange = 1
        gIter = 0

        while nChange > 0 and gIter < maxGIter:
            assGraph.initNMFVB2(None, True)
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
                                        tauType = args.tauType, biasType = args. biasType, fracCov = args.frac_cov, noiseFrac = args.noise_frac)


            assemblyGraphs = assemblyGraphsSelect
            source_maps = source_maps_select
            sink_maps = sink_maps_select

            gIter += 1
    

    #import ipdb; ipdb.set_trace()
    assGraph.initNMFVB2(None, True)

    assGraph.update(args.iters, True, None, True, logFile=args.outFileStub + "_log2.txt",drop_strain=None,relax_path=False,bMulti=True)


    assGraph.update(args.iters, True, None, True, logFile=args.outFileStub + "_log2.txt",drop_strain=None,relax_path=args.relax_path)

    assGraph.writeOutput(args.outFileStub, False, selectedSamples)

    assGraph.update(args.iters, True, None, True, logFile=args.outFileStub + "_log3.txt",drop_strain=None,relax_path=False,uncertainFactor=args.uncertain_factor)

    assGraph.update(args.iters, True, None, True, logFile=args.outFileStub + "_log3.txt",drop_strain=None,relax_path=args.relax_path)
  
    assGraph.writeOutput(args.outFileStub + "_P", False, selectedSamples)

    Gopt = assGraph.G + 1

    if (args.run_elbow and Gopt >= 5) and assGraph.S >=5:
        no_folds=int(args.nofolds)
        #no_folds2 = 2*no_folds
        
        statsH = defaultdict(list)
                  
        assErrors    = defaultdict(list)
        assMapParams = defaultdict(dict)
                
        M_attempts = 1000

        M = np.ones((assGraph.V,assGraph.S))
    

        (Ms_train1,Ms_test1) = compute_folds_attempts(prng, I=assGraph.V,J=assGraph.S,no_folds=no_folds,attempts=M_attempts,M=M)

       # (Ms_train2,Ms_test2) = compute_folds_attempts(prng, I=assGraph.V,J=assGraph.S,no_folds=no_folds,attempts=M_attempts,M=M)

        Ms_train = Ms_train1 #+ Ms_train2 
        
        Ms_test = Ms_test1 #+ Ms_test2


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
                
                prng_l = RandomState(args.random_seed + f) 
                
                pargs.append([prng_l, assemblyGraphs, source_maps, sink_maps, g, f, args, selectedSamples, outDir, M_train, M_test])            

            results = fold_p.amap(assGraphWorker,pargs)

            resultsa = list(results.get())
            
            for f in range(no_folds):
                h = resultsa[f][5]
                
                statsH[h].append((resultsa[f][0],resultsa[f][1],resultsa[f][2],resultsa[f][3],resultsa[f][4],resultsa[f][5]))
                
                assErrors[h].append((g,f,resultsa[f][4])) 
                
                assMapParams[g][f] = (resultsa[f][6], resultsa[f][7], resultsa[f][8], resultsa[f][9], resultsa[f][10], resultsa[f][11])
        
        #import ipdb; ipdb.set_trace()
        
        mean_errs = {}
        
        with open(args.outFileStub + "_CV.csv",'w') as f:
            f.write("No_strains,mean_elbo,mean_err,mean_ll,mean_ll_max,mean_ll_total\n")
            
            for h in sorted(statsH.keys()): 
            
                mean_elbo = np.mean(np.asarray([x[0] for x in statsH[h]]))        
                mean_err = np.mean(np.asarray([x[1] for x in statsH[h]])) 
                mean_ll = np.mean(np.asarray([x[2] for x in statsH[h]])) 
                mean_ll_maximal = np.mean(np.asarray([x[3] for x in statsH[h]]))
                mean_ll_total = np.mean(np.asarray([x[4] for x in statsH[h]]))
                
                mean_errs[h] = mean_ll_maximal
                
                oString = str(h) + "," + str(mean_elbo) + "," + str(mean_err) + "," + str(mean_ll) + "," + str(mean_ll_maximal) + "," + str(mean_ll_total)
                
                f.write(oString + "\n")
                
                print(oString)
    
        #Rerun with optimal g
    
        minG = max(mean_errs, key=mean_errs.get)
       
        print("Using " + str(minG) + " strains")
 
        assGraph = AssemblyPathSVA(prng, assemblyGraphs, source_maps, sink_maps, G = minG, readLength=args.readLength,
                                        ARD=True,BIAS=args.bias,  NOISE=args.NOISE, fgExePath=args.executable_path, bLoess = args.loess, bGam = args.usegam, 
                                        tauType = args.tauType, biasType = args. biasType, fracCov = args.frac_cov, noiseFrac = args.noise_frac)
   
        bestRun = max(assErrors[minG], key = lambda t: t[2])
        
        bestParams = assMapParams[bestRun[0]][bestRun[1]] 
        
        assert np.array_equal(assGraph.expGamma.shape, bestParams[0].shape)
        assGraph.expGamma = np.copy(bestParams[0])
        
        assert np.array_equal(assGraph.expGamma2.shape, bestParams[1].shape)
        assGraph.expGamma2 = np.copy(bestParams[1])
        
        assert np.array_equal(assGraph.expPhi.shape, bestParams[2].shape)
        assGraph.expPhi = np.copy(bestParams[2])
        
        assert np.array_equal(assGraph.expPhi2.shape, bestParams[3].shape)
        assGraph.expPhi2 = np.copy(bestParams[3])
        
        assert np.array_equal(assGraph.expTheta.shape, bestParams[4].shape)
        assGraph.expTheta = np.copy(bestParams[4])
        
        assert np.array_equal(assGraph.expTheta2.shape, bestParams[5].shape)
        assGraph.expTheta2 = np.copy(bestParams[5])
        
        
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
