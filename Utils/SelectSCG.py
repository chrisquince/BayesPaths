from Bio import SeqIO
from subprocess import PIPE
from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from collections import defaultdict
import os

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("scg_file", help="scgs")
        
    args = parser.parse_args()

    import ipdb; ipdb.set_trace()    


    scgs = set()
    with open(args.scg_file,'r') as scg_file:
        for line in cog_file:
            line = line.rstrip()
            
            scgs.add(line)
            
    cogFiles = glob.glob('./SR*/*_best_hits.cogs.tsv')  
    
    genomes = set()
    
    cog_genome_seq = defaultdict(dict)
    mapSeqs = {}
    
    for cogFile in cogFiles:
        genome = (os.cogFile.basename()).split('_')[0]
    
        with open(cogFile,'r') as f:
        #NODE_1_length_550379_cov_86.136683_93   COG0197 2e-67   0.5944  0.5822  0.9795  1
            for line in f:
                
                line = line.rstrip()
                
                (Query,Subject,Evalue,PID,Subject_Pid,Coverage,Query_coverage) = line.split('\t')
        
                if Subject in scgs:
                    if genome in cog_genome_seq[Subject]:
                        if Evalue < cog_genome_seq[Subject][genome][1]:
                            cog_genome_seq[Subject][genome] = (Query,Evalue)
                    else:
                        cog_genome_seq[Subject][genome] = (Query,Evalue)
        
        
        fastaFile = genome + "/" + genome + ".fna"
        
        handle = open(fastaFile, "r")
        for record in SeqIO.parse(handle, "fasta"):
            seq = record.seq

            mapSeqs[record.id] = str(seq)
        

    for scg, genome_maps in cog_genome_seq.items():
    
        scgFile = scg + ".fna" 
        
        with open(scgFile,'w') as f:
        
            for genome, map in genome_maps.items():
            
                f.write('>' + map[0] + '\n')
                
                f.write('>' + mapSeqs[map[0]] + '\n')
                
            
        


if __name__ == "__main__":
    main(sys.argv[1:])
    
    