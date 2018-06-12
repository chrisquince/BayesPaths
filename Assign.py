import sys
from Bio import SeqIO

from optparse import OptionParser
from collections import defaultdict
parser = OptionParser()
parser.add_option("-i", "--inputfile", dest="ifilename",
                  help="fasta file", metavar="FILE")


parser.add_option("-b", "--blastfile", dest="bfilename",
                  help="blast file", metavar="FILE")


(options, args) = parser.parse_args()

mapLengths = {}
mapUnitigs = {}
handle = open(options.ifilename, "rU")
for record in SeqIO.parse(handle, "fasta"):
    seq = record.seq
    mapLengths[record.id] = len(seq)
    mapUnitigs[record.id]  = []
handle.close()

#138    NC_020450.1 100.00  80  0   0   4   83  2099662 2099583 5e-36   148
#import ipdb; ipdb.set_trace()
#mapUnitigs = defaultdict(list)

with open(options.bfilename) as f:
    for l in f:
        toks = l.strip().split("\t")
        
        unitig = toks[0]
        genome = toks[1]
        pid = toks[2]

        length = toks[3]

        if float(pid) == 100. and int(length) == mapLengths[unitig]:
            mapUnitigs[unitig].append(genome)

for unitig, genomes in mapUnitigs.items():
    gString = "\t".join(genomes)
    print(unitig + "\t" + gString)        
