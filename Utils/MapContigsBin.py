#!/usr/bin/python3

import argparse
import sys
import glob

from collections import defaultdict

import numpy as np
import colorsys

import itertools
import operator

def most_common(L):
  # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index
  # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


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

    parser.add_argument("unitig_assigns", help="unitig bin assignments")

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
    maxBin = -1
 
    with open(args.unitig_assigns) as f:
        line = next(f)

        for line in f:
            line = line.strip('\n')

            toks = line.split(',')

            unitig = toks[0]
            assign = int(toks[2])

            unitigAss[unitig] = assign
#    import ipdb; ipdb.set_trace()
    for contig, unitigs in contigPaths.items():
        assigns = []
 
        for unitig in unitigs:   
            if unitig in unitigAss:
                assigns.append(unitigAss[unitig])
    
            if len(assigns) > 0:
                cass = most_common(assigns)

                print(contig + ',' + str(cass))    



if __name__ == "__main__":
    main(sys.argv[1:])



