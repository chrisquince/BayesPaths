#!/bin/bash

scripts=/mnt/gpfs/chris/Projects/STRONG_Runs/AD7_Run10/Nanopoore27/scripts

nreads=/mnt/gpfs/chris/Projects/STRONG_Runs/AD7_Run10/Nanopoore27/Reads/All_final_sana.fastq 

strongrun=/mnt/gpfs/chris/Projects/STRONG_Runs/AD7_Run10


MAG=$1

rbase=$(basename $nreads)

rbase=${rbase%.fastq}



cd $MAG

for reads in All_final_sana_${MAG}_*_filt.fasta
do
    COG=${reads%_filt.fasta}
    COG=${COG#All_final_sana_${MAG}_}

    echo $COG
    echo $reads


    GraphAligner -g ${strongrun}/subgraphs/bin_merged/${MAG}/simplif/${COG}.gfa -f $reads -a ${COG}_galign.gaf --global-alignment


done

