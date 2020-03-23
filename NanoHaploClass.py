from itertools import compress
import argparse
import sys
import numpy as np
import os
import subprocess
import re 

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
from operator import itemgetter

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
 
 
#def removeDegenerate(haplotypes,paths):


class NanoHap():

    def __init__(self,N,G,ids, unitigGraph, readLengths, readGraphMaps, dMatrix, mapID,
                    source_names, sink_names, minIter = 3, maxIter = 100):
    
        self.N = N
        
        self.G = G
        
        self.ids = ids
        
        self.unitigGraph
        
        self.readLengths = readLengths
        
        self.readGraphMaps = readGraphMaps
        
        self.dMatrix = dMatrix
        
        self.mapID = mapID
        
        self.source_names = source_names
        
        self.sink_names = sink_names
        
        self.minIter = minIter
        
        self.maxIter = maxIter
        
        self.Z = np.zeros((N,G))  
    
        self.haplotypes = {}
        
        self.paths = {}
    
        self.epsilon = 0.1
        
        self.Pi = np.zeros(self.G)
    
    def writeHaplotypes(self, fileName):
        
        with open(fileName,'w') as f:    
            for g in range(G):
                f.write(">" + str(g) + '\n')
                f.write(self.haplotypes[g] + '\n')
    
    def writePaths(self, fileName):
        pathsd = defaultdict(set)
        for g in range(G):
            for unitig in paths[g]:
                pathsd[g].add(unitig[:-1])
            
            
        with open(fileName,'w') as f:    

            for unitig in unitigGraph.unitigs:
                vals = []
                for g in range(G):
                           
                    if unitig in pathsd[g]:
                        vals.append(g) 
                    
                vString = "\t".join([str(x) for x in vals])

                f.write(unitig + "\t" + vString + "\n")
    
    
    def readAlignments(self, fileName):
    
        misMatch = defaultdict(dict)
        M = np.zeros((self.N,self.G))
        m = np.ones((self.N,self.G))
        m = m*readLengths[:,np.newaxis]
    
        with open(fileName,'r') as f:
        
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

                n = self.mapID[query]
            
                M[n,int(target)] = match
                m[n,int(target)] = mmatch
    
        return (m,M)
    
    def findHaplotypes(self):

        M, C = kMedoids(dMatrix, G)


        for g in range(G):
            ass = C[g]
        
        self.Z[ass,g] = 1. 

        deltaLL = np.finfo(float).max
        logL = -np.finfo(float).max
        lastLL = 0.0
        minChange = 1.0e-5
        
        #import ipdb; ipdb.set_trace()
        i = 0 
        while ( i < minIter or (deltaLL > minChange or i > maxIter)):
            print("iter: " + str(i))
        
            for g in range(G):
                self.unitigGraph.clearReadWeights()
    
                self.unitigGraph.setReadWeights(readGraphMaps, Z[:,g], ids)
   
                (minPath, maxSeq) = self.unitigGraph.getHeaviestBiGraphPath('readweight',source_names, sink_names)
                self.haplotypes[g] = maxSeq
                self.paths[g] = minPath
            
            self.writeHaplotypes('Haplotypes.fa')

    
            vargs = ['vsearch','--usearch_global','selectedReads.fa', '--db',
                                'Haplotypes.fa','--id','0.70','--userfields','query+target+alnlen+id+mism',
                                '--userout','hap.tsv','--maxaccepts','10']
        
            # with(open('vsearch.log','a')) as f: 
            subprocess.run(vargs,stdout=PIPE, stderr=PIPE)
         
            (m, M) = self.readAlignments('hap.tsv')
        
            mZ = np.sum(self.Z*m)
            MZ = np.sum(self.Z*M)
    
            self.epsilon = mZ/(MZ + mZ)
        
            self.Pi = np.sum(self.Z,axis=0)

            logP = np.log(self.Pi)[np.newaxis,:] + np.log(self.epsilon)*m + np.log(1.0 - self.epsilon)*M 
            lastLL = self.logL
            self.logL = 0
            
            for n in range(N):
                Z[n,:] = expNormLogProb(logP[n,:])
                Probs, dMax = expLogProb(logP[n,:])
            
                self.logL += np.log(np.sum(Probs)) + dMax
        
            if i > 0:
                deltaLL = self.logL - lastLL 
        
            print("LogLL: " + str(self.logL) + ", DeltaLL: " + str(deltaLL)) 
            
            i = i + 1




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
    np.random.seed(0)

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

            mapTok = toks[5]
            mapTok = mapTok.replace('>', '+')
            mapTok = mapTok.replace('<', '-')
            
            mapList = re.split('\+|-', mapTok)
            mapList.pop(0)
            
            dirList = re.split('[0-9]+', mapTok)
            mapList.pop()
            
            readGraphMaps[toks[0]] = [x + y for x,y in zip(mapList,dirList)]
            
    
    mapSeqs = {}
    
    handle = open(args.nanopore_reads, "r")
    for record in SeqIO.parse(handle, "fasta"):
        seq = record.seq

        mapSeqs[record.id] = str(seq)
    
    ids = list(mapSeqs.keys())
    

    (dictDist, dids) = readDistMatrix(args.var_dist_file)
    

    ids = list(set(ids).intersection(set(dids)))

    rids=list(readGraphMaps.keys())
    ids = list(set(ids).intersection(set(rids)))


    N = len(ids)
    
    readLengths = np.zeros(N)

    mapID = {ids[i]:i for i in range(N)}

    readLengths = np.asarray([len(mapSeqs[id]) for id in ids],dtype=np.int)
    
    dMatrix = np.zeros((N,N))

    for i, iid in enumerate(ids):
        for j, jid in enumerate(ids):
            dMatrix[i,j] = dictDist[iid][jid]

    

    readGraphMaps = {iid:readGraphMaps[iid] for iid in ids}

    with open('selectedReads.fa','w') as f:    
        for n in range(N):
            f.write(">" + ids[n] + '\n')
            f.write(mapSeqs[ids[n]] + '\n')

    unitigGraph.createDirectedBiGraph()

    unitigGraph.setDirectedBiGraphSource(source_names, sink_names)
    
    
    
    
    
    logLLK = defaultdict(list)
    nIters = 5
    
    for k in range(1,args.strain_number + 1):
    
        for n in range(nIters):
            np.random.seed(n)
    
            nanoHap = NanoHap(N,G,ids, unitigGraph, readLengths, readGraphMaps, dMatrix, mapID, source_names, sink_names)
        
            nanoHap.findHaplotypes()
        
            logLLK[k].append((nanoHap.logLL, nanoHap.haplotypes, nanoHap.Pi, nanoHap.epsilon))
        
            
        
    
    meanLL = {}
    for (k,listK) in logLLK.items():
        LL = [x[0] for x in listK]
    
        meanL = sum(LL)/len(LL)
        
        meanLL[k] = meanL
        
    with open('LogLL.csv','w') as f:  
        for k in sorted(meanLL.keys()):
            logLL = meanLL[k]
            f.write(str(k) + "," + str(logLL) + "\n")
    
    for k in sorted(meanLL.keys()):
        listK = logLLK[k]
        
        hapmax = max(listK,key=itemgetter(0))[1]
        pimax = max(listK,key=itemgetter(0))[2]
        
        
        with open('Haplotypes_' + str(k) + '.fa','w') as f:
            for g in range(k):  
                f.write(">" + str(g) + '\n')
                f.write(hapmax[g] + '\n')
                
        with open('Pi_' + str(k) + '.csv','w') as f:
            for g in range(k):     
                f.write(','.join([str(x) for x in pimax.tolist()]))
                f.write('\n')     
    

if __name__ == "__main__":
    main(sys.argv[1:])
