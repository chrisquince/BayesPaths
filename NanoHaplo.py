from itertools import compress
import argparse
import sys
import numpy as np
import os
import subprocess

from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

from Utils.UnitigGraph import UnitigGraph
from Utils.UtilsFunctions import convertNodeToName

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("cog_graph", help="gfa file")
    
    parser.add_argument("kmer_length", help="kmer length assumed overlap")
    
    parser.add_argument("cov_file", help="tsv file")

    parser.add_argument("nanopore_reads", help="reads")

    parser.add_argument("nanopore_maps", help="read mappings")
    
    parser.add_argument("stop_file", help="stop file")
        
    parser.add_argument("dead_end_file", help="deadend file")
    
    parser.add_argument('-g','--strain_number',nargs='?', default=5, type=int, 
        help=("number of strains"))
    
    parser.add_argument('-t','--length_list',nargs='?', default=None, help=("amino acid lengths for genes"))
    
    args = parser.parse_args()

        
    try:
        unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.cog_graph,int(args.kmer_length), args.cov_file, tsvFile=True, bRemoveSelfLinks = True)
    except IOError:
        print('Trouble using file {}'.format(args.cog_graph))
        sys.exit()
    
        
    deadEnds = []

    #import ipdb; ipdb.set_trace()

    try:
        with open(args.dead_end_file) as f:
            for line in f:
                line.strip()
                deadEnds.append(line)
    except IOError:
        print('Trouble using file {}'.format(args.dead_end_file))
        sys.exit()
        
    stops = []
        
    try:
        with open(args.stop_file) as f:
            for line in f:
                line = line.strip()
                toks = line.split("\t")
                dirn = True
                if toks[1] == '-':
                    dirn = False
                stops.append((toks[0],dirn))
    except IOError:
        print('Trouble using file {}'.format(args.stop_file))
        sys.exit()
    
    
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

        mapSeqs[record.id] = seq


    N = len(mapSeqs)
    
    ids = list(mapSeqs.keys())
    
    mapID = {ids[i]:i for i in range(N)}

    G = args.strain_number
    
    Z = np.zeros((N,G))
    
    ass = np.random.randint(G, size=N)
    
    for n,a in enumerate(ass):
        Z[n,a] = 1.
    
    unitigGraph.createDirectedBiGraph()

    unitigGraph.setDirectedBiGraphSource(source_names, sink_names)
    
    haplotypes = {}
    for g in range(G):
        unitigGraph.clearReadWeights()
    
        unitigGraph.setReadWeights(readGraphMaps, Z[:,g], ids)
   
        (minPath, maxSeq) = unitigGraph.getHeaviestBiGraphPath('readWeight',source_names, sink_names)
        haplotypes[g] = maxSeq
    
    with open('Haplotypes.fa','w') as f:    
        for g in range(G):
            f.write(">Haplo_" + str(g) + '\n')
            f.write(haplotypes[g] + '\n')
    


    vargs = ['vsearch','--usearch_global',args.nanopore_reads, '--db','Haplotypes.fa','--id','0.70','--userfields','query+target+aln+alnlen+id+mism','--userout','hap.tsv','--maxaccepts','10']
    subprocess.run(vargs)

    misMatch = defaultdict(dict)
    with open('hap.tsv','r') as f:
        
        for line in f:
            line = line.rstrip()
            toks = line.split('\t')
            
            misMatch = int(toks[5])
    
    
    M = np.zeros((N,G))

    

if __name__ == "__main__":
    main(sys.argv[1:])
