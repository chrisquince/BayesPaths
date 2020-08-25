# BayesPaths

Package for performing structured variational inference of coverage across 
multiple samples on an assembly graph.

## Prerequisites

We will require pip and setuptools for package install and Python >= 3.6. Pip can be installed with:

```
sudo apt-get update
sudo apt-get install python3-pip
```

On an Ubuntu 18.04 distribution.

## Requirements

BayesPaths uses the python packages: biopython,statsmodels, pathos, matplotlib,numpy>=1.15.4,
scipy>=1.0.0,pandas>=0.24.2,networkx>=2.4,sklearn,pygam>=0.8.0,gfapy

But these should all be installed by setup.py below.

## Installation

Download repo and install:
```
git clone https://github.com/chrisquince/BayesPaths.git
cd BayesPaths
sudo python3 ./setup.py install
```

## Quick start

We have a precomputed set of single-copy core gene graphs and coverage from the [STRONG](https://github.com/chrisquince/STRONG) pipeline 
placed in TestData.

We will run BayesPaths on a single SCG, COG0060. First create a list of COGs to run:
```
echo 'COG0504' > COG0504.txt
```

In this case just one. 

```
bayespaths TestData 77 Test504/Test504 -r 150 -g 3 -l COG0504.txt -t Data/coreCogs.tsv --nofilter -nr 1 --norun_elbow --no_ard
```

This should take 5 - 10 mins to run. This COG only contains 37 nodes too few for automatic relevance determination so we deactivated that ***--no_ard*** and 
set the number of strains to the correct number three ***-g 3***. The option ***-r 150***
sets the sequence length.
 The ***-l *** option specifies the list of COGs to run ***-t*** points to a file of COG lengths in amino acids 
which are used to help find sources and sinks on the graphs. 
The other options speed up the run  *** --nofilter -nr 1 --norun_elbow*** as a test case. See below 
for a detailed description of program arguments.

This will produce output files in the directory ***Test504*** these are also described in 
detail below but we use one the haplotype paths ***Test504/Test504F_Haplo_3_path.txt***
below to visualise the haplotypes:

```
python3 ./scripts/color_graph.py ./TestData/COG0504.gfa -p Test504/Test504F_Haplo_3_path.txt COG0504_color.gfa
```

This produces a coloured gfa ***COG0504_color.gfa*** for this COG which in [Bandage](https://rrwick.github.io/Bandage/) should appear similar to:

![alt tag](./Figures/COG0504.png)

### Full test run

A complete run of this MAG using all 35 single-copy core genes in file ***TestData/selected_cogs.tsv*** would be run as follows:

```
bayespaths TestData 77 TestData/TestData -r 150 -g 8 -l TestData/selected_cogs.tsv -t Data/coreCogs.tsv --nofilter -nr 1 --norun_elbow 
```

This will take a few hours. Here we have enabled automatic relevance determination. The program using all the COGs is able to automatically determine that three COGs are present. We have though still deactivated filtering and restricted the NMF initialisation to a single iteration ***--nofilter -nr 1***. The recommended usage on real data would be:

```
bayespaths TestData 77 TestData/TestData -r 150 -g 16 -l TestData/selected_cogs.tsv -t Data/coreCogs.tsv  
```


## Input files

## Output files

## Acknowledgements

This package uses code for a variational Bayesian NMF and cross-validation taken from the from repository (Fast Bayesian nonnegative matrix factorisation and tri-factorisation)[https://github.com/ThomasBrouwer/BNMTF] authored by Thomas Brouwer.
We have included the Apache 2.0 license from that repo here. All rights to that code reside with Thomas Brouwer.


