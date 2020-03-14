from itertools import compress
import argparse
import sys
from Bio import SeqIO

from Bio import pairwise2
from Bio.pairwise2 import format_alignment

import numpy as np

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("ref_sequence", help="alignment file")

    parser.add_argument("filtered_reads", help="fasta file")
    
    parser.add_argument("variant_file", help="variant positions")

    args = parser.parse_args()
    
    minFrac = 0.75

    #import ipdb; ipdb.set_trace()    
    
    forward = True
    handle = open(args.ref_sequence, "r")
    for record in SeqIO.parse(handle, "fasta"):
        ref_seq = record.seq
        ref_id = record.id
        if 'strand=-' in record.description:
            forward = False

    refLength = len(ref_seq)
    mapSeqs = {}
    handle = open(args.filtered_reads, "r")
    for record in SeqIO.parse(handle, "fasta"):
        seq = record.seq

        mapSeqs[record.id] = seq
    
    
    
    
    var_posns = []
    with open(args.variant_file) as variant_file:
        
        for line in variant_file:
            line = line.rstrip()
            
            toks = line.split(',')
            
            if toks[0] == ref_id:
                if forward:
                    var_posns.append(int(toks[1]))
                else:
                    revInt = refLength - int(toks[1]) - 1
                    var_posns.append(revInt)
                    
    
    var_posns.sort()
    
    nVar = len(var_posns)
    
    filtered = {}
    
    mapSeq = {'A':0, 'C':1, 'G':2, 'T':3, '-': 4}
    nF = 0
    fIds = []
    for id, seq in mapSeqs.items():
        alignments = pairwise2.align.globalxx(ref_seq, seq)

        alignE = pairwise2.align.globalms(ref_seq,seq,2, -1, -1, -1,penalize_end_gaps=False)
        
        #for a in alignE:
         #   print(format_alignment(*a))
                
        l1 = list(alignE[0][0])
        l2 = list(alignE[0][1])
        
        #trim initial reference gaps
        
        refCondense = []
        idx = 0
        
        var_posns_copy = list(var_posns)
        
        nextIdx = var_posns_copy.pop(0)
        
        for r1,r2 in zip(l1,l2):
            
            if r1 != '-':
                if idx == nextIdx:
                    refCondense.append(r2)
                    
                    if len(var_posns_copy) > 0:
                        nextIdx = var_posns_copy.pop(0)
                    else:
                        break
                
                idx += 1
            
        condS = ''.join(refCondense)
        
        nGaps = condS.count('-')

        if nGaps < minFrac*nVar:
            filtered[id] = condS
            nF += 1
            fIds.append(id)
            #print('>' + id + '\n' + condS)

    fVars = np.zeros((nF,nVar))
    for idx,fid in enumerate(fIds):
        fVars[idx,:] = np.asarray(list(map(mapSeq.get, list(filtered[fid]))))
     
    
    distMatrix = np.zeros((nF,nF))
    for c in range(nF):
        for d in range(c+1,nF):
            mask = np.ones(nVar)

            mask[fVars[c,:] == 4] = 0
            mask[fVars[d,:] == 4] = 0
    
            cVar = fVars[c,:]*mask
            dVar = fVars[d,:]*mask

            dist = np.sum(cVar != dVar)
            nComp = float(np.sum(mask))
            if nComp > 0:
                fDist = float(dist)/nComp
            else:
                fDist = 1.
            distMatrix[c,d] = fDist
            distMatrix[d,c] = fDist

    idString = ",".join(fIds)
    print('id,' + idString)
    for h in range(nF):
        hString = ",".join([str(x) for x in distMatrix[h,:].tolist()])
        
        print(fIds[h] + "," + hString)
        
        

if __name__ == "__main__":
    main(sys.argv[1:])
