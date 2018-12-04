import os
import sys
import argparse
import numpy as np
from collections import Counter,defaultdict
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


#Color_scheme=["#0277BD",]

Color_scheme=['#42A5F5','#66BB6A','#FFEB3B','#EF5350','#FF00FF']

#Color_scheme=["#F0A3FF", "#0075DC", "#993F00","#4C005C","#2BCE48","#FFCC99","#808080","#94FFB5","#8F7C00","#9DCC00","#C20088","#003380","#FFA405","#FFA8BB","#426600","#FF0010","#5EF1F2","#00998F","#740AFF","#990000","#FFFF00"]
NColors = len(Color_scheme)

def merge_color(Listcolor,List_merged):
    total_color=np.zeros(3)
    for color in Listcolor:
        total_color=total_color+np.array([int(color[1:3],16),int(color[3:5],16),int(color[5:],16)])
    int_to_hex=lambda x:hex(int(x))[2:].upper()
    Correct_int_to_hex=lambda x:int_to_hex(x)*(int_to_hex(x)!="0")*(len(int_to_hex(x))!=1)+"00"*(int_to_hex(x)=="0")+("0"+int_to_hex(x))*(int_to_hex(x)!="0")*(len(int_to_hex(x))==1)
    Merged_color="#"+"".join([Correct_int_to_hex(value) for value in total_color/len(Listcolor)])
    if len(Listcolor)>1:
        List_merged.append((tuple([Color_scheme.index(color) for color in Listcolor]),Merged_color))
    return Merged_color
    
def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("gfa_file", help="unitig fasta file in Bcalm2 format")

    parser.add_argument("strain_file", help="strain assignments tab delimited")

    parser.add_argument("strain_idx", help="strain index")

    args = parser.parse_args()

    #import ipdb; ipdb.set_trace()
    idx = int(args.strain_idx)
    mapUnitigs = {}
    set_Bugs = set([idx])
    with open(args.strain_file) as f:
        for line in f:
        
            line = line.rstrip()
        
            tokens = line.split('\t')
        
            unitig = tokens.pop(0)
            
            mapUnitigs[unitig] = [idx]
    
    list_merged=[]

    list_Bugs=sorted(list(set_Bugs))
    color_Bugs = {}
    bidx = 0
    for bug in list_Bugs:
        color_Bugs[bug] = Color_scheme[bidx % NColors + idx]
        bidx += 1
    mapUnitigColor = {}
    
    for unitig,strains in mapUnitigs.items():
        if len(strains) > 0:
            strain_colors = [color_Bugs[strain] for strain in strains]
            mapUnitigColor[unitig] = merge_color(strain_colors,list_merged)
        else:
            mapUnitigColor[unitig] = "#d3d3d3"
    
    with open(args.gfa_file) as f:
        for line in f:
            line=line.rstrip()
            toks = line.split('\t')
        
            if toks[0]=="S":
                unitig=toks[1]
                if unitig not in mapUnitigColor:
                    color="#d3d3d3"
                else:
                    color = mapUnitigColor[unitig]
                    
                toks.append("CL:z:"+color+"\tC2:z:"+color+"\n")
                
                tString = '\t'.join(toks)
                
                print(tString)
            else:
                print(line)
    

if __name__ == "__main__":
    main(sys.argv[1:])