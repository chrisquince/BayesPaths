import argparse
import sys
from UnitigGraph import UnitigGraph
from Utils import convertNodeToName
import networkx as nx
from collections import deque
import numpy as np

#class GraphProcess():
 #   """ Class for processing assembly graph"""    
    
  #  def __init__(self):

def splitComponents(unitigGraph):

    components = sorted(nx.connected_components(unitigGraph.undirectedUnitigGraph), key = len, reverse=True)

    c = 0
    for component in components:
        unitigSubGraph = unitigGraph.createUndirectedGraphSubset(component)
        
        unitigSubGraph.writeToGFA('component_' + str(c) + '.gfa')
        
        c = c + 1

def getDirected(focalGraph,u):
    
    directed = []
    
    forwardU = u + "+"
    reverseU = u + "-"
    
    if forwardU in focalGraph:
        directed.append(forwardU)
    
    if reverseU in focalGraph:
        directed.append(reverseU)
    
    return directed
    
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
                
    return (visited, sink)


def searchBubbleReverse(forwardGraph,reverseGraph,nextSink):

    for node in nx.algorithms.bfs_tree(reverseGraph, nextSink):
        if node != nextSink:
            (visited,sink) = findSuperBubble(forwardGraph,node,forwardGraph.neighbors,forwardGraph.predecessors)
            if sink is not None and sink == nextSink:
                return (node,sink,visited)

    return (None,None,None)

def bubbleOut2(forwardGraph, fullGraph, reverseFullGraph, sourceNode,sinkNode,labelledNodes):

    reverseGraph = forwardGraph.reverse()

    #get first bubble out
    out_bubbles = []
    for node in nx.algorithms.bfs_tree(reverseGraph, sourceNode):
        (visited,sink) = findSuperBubble(forwardGraph,node,forwardGraph.neighbors,forwardGraph.predecessors)
        if sink is not None:
            out_bubbles.append((node,sink,visited))
            break
    
    in_bubbles = []
    for node in nx.algorithms.bfs_tree(forwardGraph, sinkNode):
        (visited,sink) = findSuperBubble(reverseGraph,node,reverseGraph.neighbors,reverseGraph.predecessors)
        if sink is not None:
            in_bubbles.append((node,sink,visited))
            break
    
    #then add as many extra bubbles as possible
    if len(out_bubbles) > 0:
        nextSource = out_bubbles[0][1]
        while nextSource:
            (visited,sink) = findSuperBubble(forwardGraph,nextSource,forwardGraph.neighbors,forwardGraph.predecessors)
            if sink is not None:
                out_bubbles.append((nextSource,sink,visited))
            nextSource = sink
         #remove last bubble if not really a bubble
        lastBubble = out_bubbles.pop()
        (visitedTest,sinkTest) = findSuperBubble(fullGraph,lastBubble[0],fullGraph.neighbors,fullGraph.predecessors)
        if sinkTest != None:
            out_bubbles.append((lastBubble[0],sinkTest,visitedTest))
        
    if len(in_bubbles) > 0:
        nextSource = in_bubbles[0][1]
        while nextSource:
            (visited,sink) = findSuperBubble(reverseGraph,nextSource,reverseGraph.neighbors,reverseGraph.predecessors)
            if sink is not None:
                in_bubbles.append((nextSource,sink,visited))
            nextSource = sink

        #remove last bubble if not really a bubble
        lastBubble = in_bubbles.pop()
        (visitedTest,sinkTest) = findSuperBubble(reverseFullGraph,lastBubble[0],reverseFullGraph.neighbors,reverseFullGraph.predecessors)
        if sinkTest != None:
            out_bubbles.append((lastBubble[0],sinkTest,visitedTest))
        
    bubbleNodes = set()
    
    for bubble in in_bubbles + out_bubbles:
        bubbleNodes.add(bubble[1])
        bubbleNodes.add(bubble[0])
        bubbleNodes.update(bubble[2])
    
    return bubbleNodes


def bubbleOut(forwardGraph, sourceNode,labelledNodes):

    reverseGraph = forwardGraph.reverse()

    #get first bubble out
    out_bubbles = []
    for node in nx.algorithms.bfs_tree(forwardGraph, sourceNode):
        (visited,sink) = findSuperBubble(forwardGraph,node,forwardGraph.neighbors,forwardGraph.predecessors)
        if sink is not None:
            out_bubbles.append((node,sink,visited))
            break
    
    in_bubbles = []
    for node in nx.algorithms.bfs_tree(reverseGraph, sinkNode):
        (visited,sink) = findSuperBubble(forwardGraph,node,forwardGraph.neighbors,forwardGraph.predecessors)
        if sink is not None:
            in_bubbles.append((node,sink,visited))
            break
    
    #then add as many extra bubbles as possible
    if len(out_bubbles) > 0:
        nextSource = out_bubbles[0][1]
        while nextSource:
            (visited,sink) = findSuperBubble(forwardGraph,nextSource,forwardGraph.neighbors,forwardGraph.predecessors)
            if sink is not None:
                out_bubbles.append((nextSource,sink,visited))
            nextSource = sink
    
    if len(in_bubbles) > 0: 
        nextSink = in_bubbles[0][0]
    
        while nextSink:
            (source,sink,visited) = searchBubbleReverse(forwardGraph,reverseGraph,nextSink)
        
            if sink is not None:
                in_bubbles.append((source,sink,visited))
        
            nextSink = source

    bubbleNodes = set()
    
    for bubble in in_bubbles + out_bubbles:
        bubbleNodes.add(bubble[1])
        bubbleNodes.add(bubble[0])
        bubbleNodes.update(bubble[2])
    
    return bubbleNodes


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

def get_labelled_subgraph(forwardGraph, labelled):

    reverseGraph = forwardGraph.reverse()
    
    forwardReachable = set()
    reverseReachable = set()
    
    for node in labelled:
        nDes = nx.descendants(forwardGraph,node)
        nAnc = nx.descendants(reverseGraph,node)
    
        forwardReachable.update(nDes)
        
        reverseReachable.update(nAnc)    

    reachable = forwardReachable & reverseReachable
    reachable.update(labelled)
    
    return list(reachable)


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
    
def getMaximumCoverageWalk(focalGraph,bubbleNodes,covMap):
    
    sources = []
    sinks = []
    
    directedGraph = nx.DiGraph(focalGraph.subgraph(bubbleNodes))
    
    inDegree = directedGraph.in_degree(directedGraph.nodes())
            
    reachableSources = []
    for node,nDegree in inDegree:
        if nDegree == 0:
            sources.append(node)
            
    outDegree = directedGraph.out_degree(directedGraph.nodes())
            
    for node,nDegree in outDegree:
        if nDegree == 0:
            sinks.append(node)
    
    
    directedGraph.add_node('source')
    for source in sources:
        directedGraph.add_edge('source',source)
    
    directedGraph.add_node('sink')
    for sink in sinks:
        directedGraph.add_edge(sink,'sink')
    
    capacities = {}
    for e in directedGraph.edges:
        if e[1] == 'sink':
            capacities[e] = sys.float_info.max
        else:
            unitigU = e[1][:-1]
            capacities[e] = np.sum(covMap[unitigU])

    nx.set_edge_attributes(directedGraph,capacities,'capacity')
    flow_value, flow_dict = nx.maximum_flow(directedGraph, 'source', 'sink')
    
    maxWalk = []
    node = 'source'
    
    while node != 'sink':
        maxWalk.append(node)
        node = max(flow_dict[node], key=flow_dict[node].get)
    maxWalk.pop(0)
    return (flow_value, maxWalk)
    
def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("gfa_file", help="assembly graph in gfa format")

    parser.add_argument("cog_file", help="unitig cog assignments")
    
    parser.add_argument("core_cogs", help="list of core cogs")
    
    parser.add_argument("kmer_length", help="kmer length assumed overlap")
    
    parser.add_argument("outFileStub", help="stub for output files")
    
    parser.add_argument('-c','--cov_file',nargs='?', help="unitig coverages")
    
    args = parser.parse_args()

    import ipdb; ipdb.set_trace()

    if args.cov_file is not None:
        unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.gfa_file,int(args.kmer_length), args.cov_file)
    else:
        unitigGraph = UnitigGraph.loadGraphFromGfaFile(args.gfa_file,int(args.kmer_length))
    #splitComponents(unitigGraph)

    fullGraph = unitigGraph.directedUnitigBiGraph
    reverseFullGraph = fullGraph.reverse()
    coreCogs = {}
    
    with open(args.core_cogs) as f:
        for line in f:
            line = line.rstrip()
            tokens = line.split(',')
            coreCogs[tokens[0]] = float(tokens[1])

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

    covs = {}
    seqs = {}
    maxFlows = {}
    for coreCog,cogLength in coreCogs.items():
        
        g = 0
        
        coreUnitigs = deque()
        for x in unitigGraph.unitigs:
            if x in unitigCogMap and unitigCogMap[x] == coreCog:
                coreUnitigs.append(x)
        print("Total Unitigs\t" + coreCog + "\t" + str(len(coreUnitigs)))
        while len(coreUnitigs) > 0:
            coreUnitig = coreUnitigs[0]
        
            corePlusName = convertNodeToName((coreUnitig,True))
            
            focalGraph = nx.ego_graph(unitigGraph.directedUnitigBiGraph,corePlusName,radius=6.0*cogLength,center=True,undirected=True,distance='weight')
            reverseFocalGraph = focalGraph.reverse()
            
            focalGraphL = nx.ego_graph(unitigGraph.directedUnitigBiGraph,corePlusName,radius=10.0*cogLength,center=True,undirected=True,distance='weight')
            reverseFocalGraphL = focalGraphL.reverse()
            
            assert nx.is_directed_acyclic_graph (focalGraph)
            
            focalUnitigs = [n[:-1] for n in focalGraph.nodes()]
            focalCore = set(coreUnitigs).intersection(focalUnitigs)  
            focalCoreDirected = [] 
        
            for u in focalCore:
                focalCoreDirected.extend(getDirected(focalGraph,u))
        
            
            if nx.is_directed_acyclic_graph (focalGraphL):          
                extractGraph = focalGraphL
                reverseExtractGraph = reverseFocalGraphL
            else:
                extractGraph = focalGraph
                reverseExtractGraph = reverseFocalGraph
                
                
            lcd = UnitigGraph.getLowestCommonDescendant(extractGraph,focalCoreDirected)
            
            lca = UnitigGraph.getLowestCommonDescendant(reverseExtractGraph,focalCoreDirected)
                
            reachable = get_labelled_subgraph(extractGraph, lca + lcd)
            
            reachableU = [x[:-1] for x in reachable]
        
            reachableUGraph = unitigGraph.createUndirectedGraphSubset(reachableU)
        
            reachableUGraph.writeToGFA(args.outFileStub + coreCog + "R_" + str(g) + ".gfa")
            print("Graph\t" + str(g) + "\t" + str(len(focalCore)))
            for f in focalCore:
                coreUnitigs.remove(f)
        
            with open(args.outFileStub + coreCog + "R_" + str(g) + ".tsv",'w') as f:
                for x in focalCore:
                    f.write(x + "\n")
            
        
            reachableGraph = extractGraph.subgraph(reachable)
            inDegree = reachableGraph.in_degree(reachableGraph.nodes())
            
            reachableSources = []
            for node,nDegree in inDegree:
                if nDegree == 0:
                    reachableSources.append(node)
        
            outDegree = reachableGraph.out_degree(reachableGraph.nodes())
            
            reachableSinks = []
            for node,nDegree in outDegree:
                if nDegree == 0:
                    reachableSinks.append(node)        
            
            if lca is not None and lcd is not None:
                bubbleNodes = bubbleOut2(fullGraph, fullGraph, reverseFullGraph,lca[0],lcd[0],reachable)
                   
                coreFound = set([x[:-1] for x in bubbleNodes]) & focalCore
            
                print("Bubble\t" + str(g) + "\t" + str(len(coreFound)))
            
                bubbleU = [x[:-1] for x in bubbleNodes]
        
                if len(bubbleU) > 0:
                    bubbleUGraph = unitigGraph.createUndirectedGraphSubset(bubbleU)
                
                    id = coreCog + "_" + str(g)
                
                    if args.cov_file is not None:
                        (maxFlow,maxWalk) = getMaximumCoverageWalk(focalGraph,bubbleNodes,unitigGraph.covMap)
                        maxFlows[id] = maxFlow
                        print("Max flow: " + str(maxFlow))
                        contig = unitigGraph.getUnitigWalk(maxWalk)
            
                        covBubble = bubbleUGraph.computeMeanCoverage(len(contig))
                        covs[id] = covBubble
                        seqs[id] = contig
                
                    if len(coreFound) == len(focalCore):
                        bubbleUGraph.writeToGFA(args.outFileStub + coreCog + "B_" + str(g) + ".gfa")
                    
                        if args.cov_file is not None:
                            bubbleUGraph.writeCovToCSV(args.outFileStub + coreCog + "B_" + str(g) + ".csv")
                    else:
                        print("Incomplete Bubble\t" + str(g) + "\t" + str(len(focalCore) - len(coreFound)))
                        bubbleUGraph.writeToGFA(args.outFileStub + coreCog + "I_" + str(g) + ".gfa")
                    
                        if args.cov_file is not None:
                            bubbleUGraph.writeCovToCSV(args.outFileStub + coreCog + "I_" + str(g) + ".csv")
                else:
                    print("Empty Bubble\t" + str(g) + "\t" + str(len(bubbleU)))
            else:
                print("Failed to locate lca lcd\t" + str(g))
            g = g + 1
    
    if args.cov_file is not None:
        with open(args.outFileStub + 'coreContigs.fa','w') as f:
            for id, seq in seqs.items():
                f.write(">" + id + "\n")
                f.write(seq + "\n")
    
        with open(args.outFileStub + 'coreContigs_cov.csv','w') as f:
            for id, covs in covs.items():
                f.write(id + ",")
                cString = ",".join([str(x) for x in covs.tolist()])
                f.write(cString + "\n")
    
        with open(args.outFileStub + 'maxFlow.csv','w') as f:
            for id, maxFlow in maxFlows.items():
                f.write(id + "," + str(maxFlow) + "\n")
            
    
    print("Debug")
        #    for u in coreGraphUnitigsU:
         #       f.write(u + '\t0\n')
    
        
    import ipdb; ipdb.set_trace()
        #focalGraph = get_hairy_ego_graph(unitigGraph.directedUnitigBiGraph,corePlusName,5000,'weight')
        
        
     #   coreGraphU = [x[:-1] for x in coreGraph]
        
      #  coreUGraph = unitigGraph.createUndirectedGraphSubset(coreGraphU)
        
       # coreUGraph.writeToGFA(coreCog + ".gfa")
        
        #coreGraphUnitigs = []
        
        #for x in coreGraph.nodes():
         #   if x[:-1] in coreUnitigs:
          #      coreGraphUnitigs.append(x)
        
        #coreGraphUnitigsU = [x[:-1] for x in coreGraphUnitigs]
        

        
        
       # reachableCore = UnitigGraph.getReachableSubset(coreGraph,coreGraphUnitigs)
       # reachableCoreU = [x[:-1] for x in reachableCore]
        
        #coreUGraph = unitigGraph.createUndirectedGraphSubset(reachableCoreU)
        
        #coreUGraph.writeToGFA(coreCog + "R.gfa")
        
        #print("Debug")

    
if __name__ == "__main__":
    main(sys.argv[1:])