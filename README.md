# BayesPaths

Package for performing structured variational inference of coverage across 
multiple samples on an assembly graph.

## Requirements

Python >= 3.5, Python packages: matplotlib,numpy>=1.15.4,scipy>=1.0.0",pandas>=0.24.2,networkx>=2.4,sklearn,pygam>=0.8.0,gfapy

## Installation

Download repo and install:
```
git clone https://github.com/chrisquince/BayesPaths.git
cd BayesPaths
sudo python3 ./setup.py install
```

## Quick start

We have a precomputed set of single-copy core gene graphs and coverage from the STRONG pipeline 
placed in TestData.

We will run BayesPaths on a single SCG, COG0060. First create a list of COGs to run:
```
echo 'COG0504' > COG0504.txt
```

In this case just one. 

```
mkdir Test504
bayespaths TestData 77 Test504/Test504 -r 150 -g 8 -l COG0504.txt -t Data/coreCogs.tsv --nofilter -nr 1 --norun_elbow 
```

```
sed 's/COG0504_//g' Test504maxPath.tsv > Test504.tsv
python3 ../Utils/Add_color.py ../TestData/COG0504.gfa Test504.tsv > COG0504_color.gfa
```

## Input files

## Output files
