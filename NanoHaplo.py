from itertools import compress
import argparse
import sys
import numpy as np


from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

from Utils.UnitigGraph import UnitigGraph

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("cog_graph", help="gfa file")
    
    parser.add_argument("kmer_length", help="kmer length assumed overlap")
    
    parser.add_argument("cov_file", help="tsv file")

    parser.add_argument("nanopore_reads", help="reads")

    parser.add_argument("nanopore_maps", help="read mappings")
    
    args = parser.parse_args()

        
    try:
        unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.cog_graph,int(args.kmer_length), args.cov_file, tsvFile=True, bRemoveSelfLinks = True)
    except IOError:
        print('Trouble using file {}'.format(args.cog_graph))
        
        sys.exit()
        
    


if __name__ == "__main__":
    main(sys.argv[1:])