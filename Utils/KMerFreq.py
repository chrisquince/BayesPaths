import gzip
import sys
import argparse
import re
import logging

import numpy as np
import pandas as p

from itertools import product, tee
from collections import Counter, OrderedDict

from Bio import SeqIO

def generate_feature_mapping(kmer_len):
    BASE_COMPLEMENT = {"A":"T","T":"A","G":"C","C":"G"}
    kmer_hash = {}
    counter = 0
    for kmer in product("ATGC",repeat=kmer_len):
        if kmer not in kmer_hash:
            kmer_hash[kmer] = counter
            rev_compl = tuple([BASE_COMPLEMENT[x] for x in reversed(kmer)])
            kmer_hash[rev_compl] = counter
            counter += 1
    return kmer_hash, counter

def window(seq,n):
    els = tee(seq,n)
    for i,el in enumerate(els):
        for _ in range(i):
            next(el, None)
    return zip(*els)

def _calculate_composition(read_file, kmer_len, length_threshold=25):
    #Generate kmer dictionary
    feature_mapping, nr_features = generate_feature_mapping(kmer_len)
    composition = np.zeros(nr_features,dtype=np.int)
    start_composition = np.zeros(nr_features,dtype=np.int)
    
    with gzip.open(read_file, "rt") as handle:

        for seq in SeqIO.parse(handle,"fastq"):
            seq_len = len(seq)
            if seq_len<= length_threshold:
                continue
            str_seq = str(seq.seq)
            # Create a list containing all kmers, translated to integers
            kmers = [
                    feature_mapping[kmer_tuple]
                    for kmer_tuple 
                    in window(str_seq.upper(), kmer_len)
                    if kmer_tuple in feature_mapping
                    ]
            # numpy.bincount returns an array of size = max + 1
            # so we add the max value and remove it afterwards
            # numpy.bincount was found to be much more efficient than
            # counting manually or using collections.Counter
            kmers.append(nr_features - 1)
            composition_v = np.bincount(np.array(kmers))
            composition_v[-1] -= 1
            # Adding pseudo counts before storing in dict
            composition += composition_v
            failStart = 0
            if seq_len >= kmer_len:
                startKmer = str_seq[0:kmer_len].upper()
                startKmerT = tuple(startKmer)
                if startKmerT in feature_mapping:
                    start_composition[feature_mapping[startKmerT]]+=1 
                else:
                    failStart+=1
        
        return feature_mapping, composition, start_composition, failStart


def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("read_file", help="gzipped fastq read file")
    
    parser.add_argument("kmer_length", help="kmer length assumed overlap")
    
    parser.add_argument("outFileStub", help="stub for output files")
    
    args = parser.parse_args()

    #import ipdb; ipdb.set_trace()
    
    (feature_mapping, composition, start_composition,failStart) = _calculate_composition(args.read_file, int(args.kmer_length))

    print(str(failStart))
    for k in sorted(feature_mapping, key=feature_mapping.get):
        kidx = feature_mapping[k]
        print("".join(k) + "," + str(kidx) + "," + str(composition[kidx]) + "," + str(start_composition[kidx]) )

if __name__ == "__main__":
    main(sys.argv[1:])
