from itertools import compress
import argparse
import sys
import numpy as np
import os
import subprocess

from subprocess import PIPE
from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from collections import defaultdict

from Utils.UnitigGraph import UnitigGraph
from Utils.UtilsFunctions import convertNodeToName
from Utils.UtilsFunctions import expNormLogProb
from Utils.UtilsFunctions import expLogProb
from kmedoids.kmedoids import kMedoids


def readDistMatrix(var_dist_file):

    first = True
    dids = []
    dictDist = defaultdict(dict)
    with open(var_dist_file) as f:
        for line in f:
    
            if first:
                line = line.rstrip()

                toks = line.split(',')

                toks.pop(0)

                dids = toks

                first = False
            else:
                line = line.rstrip()

                toks = line.split(',')                
                did = toks.pop(0)

                for d, djd in enumerate(dids):
                    dictDist[did][djd] = float(toks[d])

    return (dictDist, dids)
    
def readCogStopsDead(cog_graph,kmer_length,cov_file):

    deadEndFile = cog_graph[:-3] + "deadends"
        
    stopFile = cog_graph[:-3] + "stops"

    try:
        unitigGraph = UnitigGraph.loadGraphFromGfaFile(cog_graph,int(kmer_length), cov_file, tsvFile=True, bRemoveSelfLinks = True)
    except IOError:
        print('Trouble using file {}'.format(cog_graph))
        sys.exit()
    
        
    deadEnds = []


    try:
        with open(deadEndFile) as f:
            for line in f:
                line.strip()
                deadEnds.append(line)
    except IOError:
        print('Trouble using file {}'.format(deadEndFile))
        sys.exit()
        
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
        sys.exit()
        
    return (unitigGraph, stops, deadEnds )
 
def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("cog_graph", help="gfa file")
    
    parser.add_argument("kmer_length", help="kmer length assumed overlap")
    
    parser.add_argument("cov_file", help="tsv file")

    parser.add_argument("nanopore_reads", help="reads")

    parser.add_argument("nanopore_maps", help="read mappings")
    
    parser.add_argument("var_dist_file", help="variant distances for Nanopore reads")

    parser.add_argument('-g','--strain_number',nargs='?', default=5, type=int, 
        help=("number of strains"))
    
    parser.add_argument('-t','--length_list',nargs='?', default=None, help=("amino acid lengths for genes"))
    
    args = parser.parse_args()

    import ipdb; ipdb.set_trace()    

    (unitigGraph, stops, deadEnds ) = readCogStopsDead(args.cog_graph,args.kmer_length,args.cov_file)
    
    cogLengths = {}
    if  args.length_list != None:
        with open(args.length_list,'r') as cog_file:
            for line in cog_file:
                line = line.rstrip()
                toks = line.split('\t') 
                cogLengths[toks[0]] = float(toks[1])
    
    gene = os.path.splitext(os.path.basename(args.cog_graph))[0] 
    if gene in cogLengths:
        (source_list, sink_list) = unitigGraph.selectSourceSinksStops(stops, deadEnds, cogLengths[gene]*3)
    else:
        (source_list, sink_list) = unitigGraph.selectSourceSinksStops(stops, deadEnds)
    
    source_names = [convertNodeToName(source) for source in source_list] 
    sink_names = [convertNodeToName(sink) for sink in sink_list]
        
    readGraphMaps = {}
    
    with open(args.nanopore_maps) as f:
        for line in f:
            line = line.rstrip()
            
            toks = line.split('\t')
            
            readGraphMaps[toks[0]] = toks[6].split(',')
            
    
    mapSeqs = {}
    
    handle = open(args.nanopore_reads, "r")
    for record in SeqIO.parse(handle, "fasta"):
        seq = record.seq

        mapSeqs[record.id] = str(seq)
    
    ids = list(mapSeqs.keys())
    

    (dictDist, dids) = readDistMatrix(args.var_dist_file)
    

    ids = list(set(ids).intersection(set(dids)))

    N = len(ids)
    
    readLengths = np.zeros(N)

    mapID = {ids[i]:i for i in range(N)}

    readLengths = np.asarray([len(mapSeqs[id]) for id in ids],dtype=np.int)

    G = args.strain_number
    
    Z = np.zeros((N,G))    
    
    dMatrix = np.zeros((N,N))

    for i, iid in enumerate(ids):
        for j, jid in enumerate(ids):
            dMatrix[i,j] = dictDist[iid][jid]

    readGraphMaps = {iid:readGraphMaps[iid] for iid in ids}

    with open('selectedReads.fa','w') as f:    
        for n in range(N):
            f.write(">" + ids[n] + '\n')
            f.write(mapSeqs[ids[n]] + '\n')


    M, C = kMedoids(dMatrix, G)

    for g in range(G):
        ass = C[g]
        
        Z[ass,g] = 1. 
    
    unitigGraph.createDirectedBiGraph()

    unitigGraph.setDirectedBiGraphSource(source_names, sink_names)
    
    maxIter = 100

    deltaLL = np.finfo(float).max
    logL = -np.finfo(float).max
    lastLL = 0.0
    minChange = 1.0e-5
    #import ipdb; ipdb.set_trace()
    i = 0 
    while ( i < 10 or deltaLL > minChange):
        print("iter: " + str(i))
        haplotypes = {}
        for g in range(G):
            unitigGraph.clearReadWeights()
    
            unitigGraph.setReadWeights(readGraphMaps, Z[:,g], ids)
   
            (minPath, maxSeq) = unitigGraph.getHeaviestBiGraphPath('readWeight',source_names, sink_names)
            haplotypes[g] = maxSeq
    
        with open('Haplotypes.fa','w') as f:    
            for g in range(G):
                f.write(">" + str(g) + '\n')
                f.write(haplotypes[g] + '\n')
    


        vargs = ['vsearch','--usearch_global','selectedReads.fa', '--db',
                    'Haplotypes.fa','--id','0.70','--userfields','query+target+alnlen+id+mism',
                                    '--userout','hap.tsv','--maxaccepts','10']
        
       # with(open('vsearch.log','a')) as f: 
        subprocess.run(vargs,stdout=PIPE, stderr=PIPE)
        
        #import ipdb; ipdb.set_trace()
        misMatch = defaultdict(dict)
        M = np.zeros((N,G))
        m = np.ones((N,G))
        m = m*readLengths[:,np.newaxis]
    
        with open('hap.tsv','r') as f:
        
            for line in f:
                line = line.rstrip()
                toks = line.split('\t')
                
                query = toks[0]
                target = toks[1]
                alen = int(toks[2])
                pid = float(toks[3])/100.0
            
                match  = int(pid*alen)
                mmatch = int((1-pid)*alen)
            
                misMatch[query][target]= (match,mmatch)

                n = mapID[query]
            
                M[n,int(target)] = match
                m[n,int(target)] = mmatch
         

        
        mZ = np.sum(Z*m)
        MZ = np.sum(Z*M)
    
        epsilon = mZ/(MZ + mZ)
        #epsilon = 0.05
        Pi = np.sum(Z,axis=0)
        print(epsilon)
        print(Pi.tolist())
        logP = np.log(Pi)[np.newaxis,:] + np.log(epsilon)*m + np.log(1.0 - epsilon)*M 
        lastLL = logL
        logL = 0
        for n in range(N):
            Z[n,:] = expNormLogProb(logP[n,:])
            Probs, dMax = expLogProb(logP[n,:])
            
            logL += np.log(np.sum(Probs)) + dMax
        
        if i > 0:
            deltaLL = logL - lastLL 
        
        print("LogLL: " + str(logL) + ", DeltaLL: " + str(deltaLL)) 
        
        i = i + 1

if __name__ == "__main__":
    main(sys.argv[1:])
