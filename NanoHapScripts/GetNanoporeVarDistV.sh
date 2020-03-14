#!/bin/bash

scripts=/mnt/gpfs/chris/Projects/STRONG_Runs/AD7_Run10/Nanopoore27/scripts

nreads=/mnt/gpfs/chris/Projects/STRONG_Runs/AD7_Run10/Nanopoore27/Reads/All_final_sana.fastq 

strongrun=/mnt/gpfs/chris/Projects/STRONG_Runs/AD7_Run10


MAG=$1

rbase=$(basename $nreads)

rbase=${rbase%.fastq}

echo $rbase

mkdir ${MAG}

count=0
while read cog length
do 

    minlen=$(bc <<< ${length}/1)

    echo $length
    echo $minlen

    cog_count=$(grep -c "${cog}" ${strongrun}/bayespaths/$MAG/${MAG}*Q*fa)
    
    echo $cog_count

    if [ $cog_count -gt 0 ]; then

        grep "${cog}" ${strongrun}/bayespaths/$MAG/${MAG}*Q*fa -A 1 | sed 's/-//g' | sed '/^$/d' > ${MAG}/${MAG}_${cog}.fa 

        grep $cog -A 1 ${strongrun}/subgraphs/bin_merged/${MAG}/SCG.fna > ${MAG}/${cog}.fa

        (minimap2 -cx map-ont -t16 ${MAG}/${MAG}_${cog}.fa $nreads > ${MAG}/${rbase}_${MAG}_${cog}.paf;
        python3 ${scripts}/ParseSeq.py $nreads ${MAG}/${rbase}_${MAG}_${cog}.paf -l $minlen > ${MAG}/${rbase}_${MAG}_${cog}_filt.fasta;
        python3 ${scripts}/CondenseSeq.py  ${MAG}/${cog}.fa ${MAG}/${rbase}_${MAG}_${cog}_filt.fasta ${strongrun}/desman/${MAG}/freqs_sel_var.csv > ${MAG}/${rbase}_${MAG}_${cog}_filt.vdist;
        cat ${MAG}/${MAG}_${cog}.fa ${MAG}/${rbase}_${MAG}_${cog}_filt.fasta > ${MAG}/${rbase}_${MAG}_${cog}_filt_ref.fasta;
        python3 ${scripts}/CondenseSeq.py  ${MAG}/${cog}.fa ${MAG}/${rbase}_${MAG}_${cog}_filt_ref.fasta ${strongrun}/desman/${MAG}/freqs_sel_var.csv > ${MAG}/${rbase}_${MAG}_${cog}_filt_ref.vdist)&

    fi

    if [ $count -eq 8 ]; then
            wait
    fi
    
    let count=count+1

done < 'coreCogsR.tsv'




