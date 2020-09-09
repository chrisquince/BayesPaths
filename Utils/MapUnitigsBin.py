#!/usr/bin/python3

import argparse
import sys
import glob

from collections import defaultdict

import numpy as np
import colorsys

def get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        
        fColor = ''.join('{:02X}'.format(int(255*a)) for a in rgb) 

        colors.append(fColor)
    return colors


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("contig_paths", help="contig unitig path")

    parser.add_argument("contig_assigns", help="contig bin assignments")

    parser.add_argument("contig_covs", help="contig coverages")
    
    args = parser.parse_args()
#    import ipdb; ipdb.set_trace()
    

    contigPaths = defaultdict(set)
    with open(args.contig_paths) as f:
        content = f.readlines()

        content = [x.strip() for x in content] 

        for idx, line in enumerate(content):    
            if line.startswith("NODE"):
                contig = line.strip('\'')
            else:
                toks = line.split(',')

                for tok in toks:
                    tok = tok.strip('+-;')
                    contigPaths[contig].add(tok)

    unitigAss = {}
    unitigCov = {}
    maxBin = -1

    
    contigCovs = {}
    with open(args.contig_covs) as f:
        line = next(f)
        headerc = line
        for line in f:
            line = line.strip('\n')

            toks = line.split('\t')
            cString = '\t'.join(toks[1:])
            contigCovs[toks[0]] = cString


    for contig, paths in contigPaths.items():
        for unitig in paths:
            unitigCov[unitig] = contigCovs[contig] 

    with open(args.contig_assigns) as f:
        line = next(f)

        for line in f:
            line = line.strip('\n')

            toks = line.split(',')

            contig = toks[0]
            assign = int(toks[1])

            if contig in contigPaths:
                for unitig in contigPaths[contig]:
                    if assign > maxBin:
                        maxBin = assign
                    
                    unitigAss[unitig] = assign
 #                   unitigCov[unitig] = contigCovs[contig] 
                   #print(unitig + ',' + str(assign))
    colors = get_colors(maxBin + 1)

    for unitig, ass in unitigAss.items():

        print(unitig + "," + colors[ass] + "," + str(ass))
    
    with open('unitig_cov.tsv','w') as f:
        f.write(headerc)
        for unitig, cov in unitigCov.items():
            f.write(unitig + "\t" + cov + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])



