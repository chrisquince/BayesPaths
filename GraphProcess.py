import argparse
import sys
from UnitigGraph import UnitigGraph
from Utils import convertNodeToName
import networkx as nx

#class GraphProcess():
 #   """ Class for processing assembly graph"""    
    
  #  def __init__(self):
    
    

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("gfa_file", help="assembly graph in gfa format")

    parser.add_argument("cog_file", help="unitig cog assignments")
    
    parser.add_argument("core_cogs", help="list of core cogs")
    
    parser.add_argument("cov_file", help="unitig coverages")
    
    parser.add_argument("kmer_length", help="kmer length assumed overlap")
    
    args = parser.parse_args()

    import ipdb; ipdb.set_trace()

    coreCogs = set()
    
    with open(args.core_cogs) as f:
        for line in f:
            line = line.rstrip()
    
            coreCogs.add(line)

    unitigCogMap = {}
    with open(args.cog_file) as f:
        for line in f:
        
            line = line.rstrip()
        
            tokens = line.split('\t')
        
            unitig = tokens[0]
            
            cog = tokens[1]
            
            if cog in coreCogs:
                unitigCogMap[unitig] = cog
            
            
    unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.gfa_file,int(args.kmer_length), args.cov_file)
    
    for coreCog in coreCogs:
        coreUnitigs = []
        for x in unitigGraph.unitigs:
            if x in unitigCogMap and unitigCogMap[x] == coreCog:
                coreUnitigs.append(x)
        
        coreUnitig = coreUnitigs[0]
        
        corePlusName = convertNodeToName((coreUnitig,True))
        
        coreGraph = nx.ego_graph(unitigGraph.directedUnitigBiGraph,corePlusName,undirected=True,radius=5000,distance='weight')
        
        coreGraphU = [x[:-1] for x in coreGraph.nodes()]
        
        coreUGraph = unitigGraph.createUndirectedGraphSubset(coreGraphU)
        
        coreUGraph.writeToGFA(coreCog + ".gfa")
        
        coreGraphUnitigs = []
        
        for x in coreGraph.nodes():
            if x[:-1] in coreUnitigs:
                coreGraphUnitigs.append(x)
        
        coreGraphUnitigsU = [x[:-1] for x in coreGraphUnitigs]
        
        with open('coreU.txt','w') as f:
            for u in coreGraphUnitigsU:
                f.write(u + '\t0\n')
        
        
        reachableCore = UnitigGraph.getReachableSubset(coreGraph,coreGraphUnitigs)
        reachableCoreU = [x[:-1] for x in reachableCore]
        
        coreUGraph = unitigGraph.createUndirectedGraphSubset(reachableCoreU)
        
        coreUGraph.writeToGFA(coreCog + "R.gfa")
        
        print("Debug")


    #get separate components in graph
    #components = sorted(nx.connected_components(unitigGraph.undirectedUnitigGraph), key = len, reverse=True)

    #c = 0
    #for component in components:
     #   unitigSubGraph = unitigGraph.createUndirectedGraphSubset(component)
        
      #  unitigSubGraph.writeToGFA('component_' + str(c) + '.gfa')
        
       # c = c + 1
    
    #import ipdb; ipdb.set_trace()
    
if __name__ == "__main__":
    main(sys.argv[1:])