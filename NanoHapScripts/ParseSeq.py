import argparse
import sys
from Bio import SeqIO


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("query_seq", help="fastq query reads")

    parser.add_argument("paf_file", help="kmer length assumed overlap")

    parser.add_argument('-l','--min_length',nargs='?', type=int, default=1000)

    args = parser.parse_args()

#    import ipdb; ipdb.set_trace() 

    minqual = -1
    minlength = args.min_length
    minid = 0.8

    mapHits = {}
    with open(args.paf_file,'r') as f:
        for line in f:
            toks = line.strip().split('\t')

            query_name = toks[0]

            query_start = int(toks[2])

            query_end = int(toks[3])

            strand = toks[4]

            matches =  int(toks[9])

            align_len = int(toks[10])

            quality = int(toks[11])

            pid = float(matches)/float(align_len)

            if pid > minid and quality > minqual:
              #  print(str(pid))
                if align_len > minlength:
                    mapHits[query_name] = (query_start,query_end,strand,minlength,pid)

    handle = open(args.query_seq, "rU")
    for record in SeqIO.parse(handle, "fastq"):
        seq = record.seq

        if record.id in mapHits:
            print('>' + record.id)
            
            frag = seq[mapHits[record.id][0]:mapHits[record.id][1]]
            
            if mapHits[record.id][2]  == '-':
                frag = frag.reverse_complement()
    
            print(frag)
    
    handle.close()

if __name__ == "__main__":
    main(sys.argv[1:])
