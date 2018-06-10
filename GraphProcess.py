import argparse
import sys
from UnitigGraph import UnitigGraph
from Utils import convertNodeToName
import networkx as nx
from collections import deque
#class GraphProcess():
 #   """ Class for processing assembly graph"""    
    
  #  def __init__(self):
    
def generic_hairy_ego_graph(graph,source,radius,metric,neighbors,distanceSource):
    sinks = []
    visited = {source}
    
    queue = deque([(source, neighbors(source))]) 
    distance = {}
    distance[source] = 0.0
    
    while queue:
        parent, children = queue[0]
        try:
            child = next(children)
            
            if child not in visited:
                visited.add(child)
                
                if distanceSource[child] > radius and len(list(neighbors(child))) == 1:
                    sinks.append(child)
                else:
                    queue.append((child, neighbors(child)))
        except StopIteration:
            queue.popleft()
    return sinks

    
def findSuperBubble(directedGraph, source, descendents, parents):

    queue = deque([source])
    
    seen = set()
    visited = set()
    
    sink = None
    
    while queue:
        node = queue.popleft()
        
        visited.add(node)
        if node in seen:
            seen.remove(node)
            
        nList = list(descendents(node))
        
        if len(nList) > 0:
            for child in nList:
                if child == source:
                    break
                
                seen.add(child)
                         
                childVisited = set(parents(child))

                if childVisited.issubset(visited): 
                    queue.append(child)
        
        if len(queue) == 1 and len(seen) < 2:
            if len(seen) == 0 or queue[0] in seen:
                if not directedGraph.has_edge(queue[0],source):
                    sink = queue[0]
                    break
                
    return sink
    

def generic_ego_graph(graph,source,radius,metric,neighbors,distanceSource):
    sinks = []
    visited = {source}
    
    queue = deque([(source, neighbors(source))]) 
    distance = {}
    distance[source] = 0.0
    
    while queue:
        parent, children = queue[0]
        try:
            child = next(children)
            
            if child not in visited:
                visited.add(child)
                
                if distanceSource[child] > radius:
                    sinks.append(child)
                else:
                    queue.append((child, neighbors(child)))
        except StopIteration:
            queue.popleft()
    return (visited,sinks)



def get_hairy_ego_graph(directedGraph,origin,radius,metric):

    successors = directedGraph.neighbors
    
    distanceSourceF = nx.shortest_path_length(directedGraph, source=origin,weight=metric)
    
    sinks = generic_hairy_ego_graph(directedGraph,origin,radius,metric,successors,distanceSourceF)

    directedGraphR = directedGraph.reverse(copy=True)
    distanceSourceR = nx.shortest_path_length(directedGraphR, source=origin,weight=metric)
    
    successors = directedGraph.predecessors
    
    sources = generic_hairy_ego_graph(directedGraph,origin,radius,metric,successors,distanceSourceR)
 
    sink_reachable = set()
    for sink in sinks:
        #sink_reachable.add(sink)
        distanceSink = nx.shortest_path_length(directedGraphR, source=sink,weight=metric)
        (sink_reach,sink_sink) = generic_ego_graph(directedGraphR,sink,radius,metric,directedGraphR.neighbors,distanceSink)    
        sink_reachable |= set(sink_reach)
    
    source_reachable = set()
    for source in sources:
        #source_reachable.add(source)
        distanceSource = nx.shortest_path_length(directedGraph, source=source,weight=metric)
        (source_reach,source_sink) = generic_ego_graph(directedGraph,source,radius,metric,directedGraph.neighbors,distanceSource) 
        source_reachable |= set(source_reach)
    
    reachable = sink_reachable & source_reachable
    
    return list(reachable)


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("gfa_file", help="assembly graph in gfa format")

    parser.add_argument("cog_file", help="unitig cog assignments")
    
    parser.add_argument("core_cogs", help="list of core cogs")
    
    parser.add_argument("cov_file", help="unitig coverages")
    
    parser.add_argument("kmer_length", help="kmer length assumed overlap")
    
    args = parser.parse_args()

    import ipdb; ipdb.set_trace()

    unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.gfa_file,int(args.kmer_length), args.cov_file)

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
            
    
    #import ipdb; ipdb.set_trace()

    
    for coreCog in coreCogs:
        coreUnitigs = []
        for x in unitigGraph.unitigs:
            if x in unitigCogMap and unitigCogMap[x] == coreCog:
                coreUnitigs.append(x)
        
        coreUnitig = coreUnitigs[0]
        
        corePlusName = convertNodeToName((coreUnitig,True))
        
        #start at focal node and move down graph finding bubbles
        corePlusName = '221616044-'
        
        coreSink = findSuperBubble(unitigGraph.directedUnitigBiGraph, corePlusName,unitigGraph.directedUnitigBiGraph.neighbors,unitigGraph.directedUnitigBiGraph.predecessors)
        
        
        coreGraphU = [x[:-1] for x in coreGraph]
        
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

    components = sorted(nx.connected_components(unitigGraph.undirectedUnitigGraph), key = len, reverse=True)

    c = 0
    for component in components:
        unitigSubGraph = unitigGraph.createUndirectedGraphSubset(component)
        
        unitigSubGraph.writeToGFA('component_' + str(c) + '.gfa')
        
        c = c + 1

    
if __name__ == "__main__":
    main(sys.argv[1:])