import argparse
import sys
import glob
import numpy as np
import os

from UnitigGraph import UnitigGraph
from AssemblyPathSVA import AssemblyPathSVA
from Utils import convertNodeToName
from numpy.random import RandomState
from collections import defaultdict

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0) + b - a)

def overlap(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    overlap = np.sum(np.minimum(a,b))

    return min(np.sum(a),np.sum(b)) - overlap

def overlapDist(gammaMatrixG, gammaMatrixH):

    dist = 0.
    
    G = gammaMatrixG.shape[0]
    H = gammaMatrixH.shape[0]
    S = gammaMatrixH.shape[1]
    
    GCopy = np.sum(gammaMatrixG)
    
    GSum = np.sum(gammaMatrixG,axis=1)
    
    gSort = np.argsort(GSum):
    
    assigned = np.full(H,-1)
    totalDist = 0.
    for gidx in gSort:
        minDist = sys.float_info.max
        minH = -1
        for h in range(H):
            if assigned[h] < 0:    
                distGH = overlap(GCopy[gidx,:],gammaMatrixH[h,:])
                if distGH < minDist:
                    minDist = distGH
                    minH = h
        if minH >= 0:
            assigned[minH] = gidx
            totalDist += minDist
            GCopy[gidx,:] = np.maximum(GCopy[gidx,:] - gammaMatrixH[minH],0)
    
    for h in range(H):
        if assigned[h] < 0.:
            totalDist += np.sum(gammaMatrixH[h,:])
    
    return (totalDist,totalDist/np.sum(gammaMatrixH))
def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("Gene_dir", help="directory with gfa files in")

    parser.add_argument("kmer_length", help="kmer length assumed overlap")

    parser.add_argument("outFileStub", help="output file stub")

    parser.add_argument('-g','--strain_number',nargs='?', default=5, type=int, 
        help=("maximum number of strains"))

    parser.add_argument('-f','--frac',nargs='?', default=0.75, type=float, 
        help=("fraction for path source sink"))

    args = parser.parse_args()

    import ipdb; ipdb.set_trace()
    
    np.random.seed(2)
    prng = RandomState(238329)
    
    gfaFiles = glob.glob(args.Gene_dir + '/*.gfa')    

    assemblyGraphs = defaultdict(list)
    sink_maps = defaultdict(list)
    source_maps = defaultdict(list)
    idx_map = {}
    reverse_map = {}
    genes = set()
    stubs = []
    for gfaFile in gfaFiles:
        fileName = os.path.basename(gfaFile)
        
        toks = fileName.split('_')
        
        gene = toks[1][:-1]
        
        idx = toks[2]
        
        #comp1R_COG0090B_1.csv
        stub = gfaFile[:-3]
        stubs.append(stub)
        covFile = stub + "csv"
        
        unitigGraph = UnitigGraph.loadGraphFromGfaFile(gfaFile,int(args.kmer_length), covFile)
            
        (source_list, sink_list) = unitigGraph.selectSourceSinks2(args.frac)

        source_names = [convertNodeToName(source) for source in source_list] 
        sink_names = [convertNodeToName(sink) for sink in sink_list]
            
        sink_maps[gene].append(sink_list)
        source_maps[gene].append(source_list)
        assemblyGraphs[gene].append(unitigGraph)
        idx_map[stub] = (gene,idx)
        reverse_map[(gene,idx)] = stub
        genes.add(gene)


    strains = []
    thresh = 1.0
    
    assGraphs = {}
    for gene in sorted(genes):
        idx = 0
        for (sink_map,source_map,assemblyGraph) in zip(sink_maps[gene],source_maps[gene],assemblyGraphs[gene]):
            assGraph = AssemblyPathSVA(prng, assemblyGraph, source_map, sink_map, G = args.strain_number, readLength=150,ARD=True)
    
            assGraph.initNMF()

            assGraph.update(100, True)
            
            stub = reverse_map[(gene,idx)]
            assGraphs[stub] = assGraph
            
            stubFile = args.outFileStub + stub
            
            assGraph.writeMarginals(stubFile + "margFile.csv")
   
            assGraph.getMaximalUnitigs(stubFile + "Haplo_" + str(assGraph.G) + ".fa")
 
            assGraph.writeMaximals(stubFile + "maxFile.tsv")
   
            assGraph.writeGammaMatrix(stubFile + "Gamma.csv") 
            
            idx = idx + 1
    

    distStubs = defaultdict(dict)
    
    for stubI in stubs:
        for stubJ in stubs:
        
            distStubs[distI][distJ] = overlapDist(assGraphs[stubI].expGamma, assGraphs[stubJ].expGamma)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
