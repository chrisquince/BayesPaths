import argparse
import sys
from Utils.UnitigGraph import UnitigGraph
from Utils.UtilsFunctions import convertNodeToName
import networkx as nx
from collections import deque
from collections import defaultdict
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

def add_connecting_nodes(reachable, forwardGraph, labelled):

    augmented = set(reachable)
    
    for node in reachable:
        if node not in labelled:
            for neighbor in forwardGraph.neighbors(node):
                augmented.add(neighbor)
                
            for neighbor in forwardGraph.predecessors(node):
                augmented.add(neighbor)
                
    return list(augmented)


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
    
    
    if len(sources) > 0 and len(sinks) > 0:
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
    
        bProblem = False
        while node != 'sink':
            maxWalk.append(node)
            if len(flow_dict[node]) > 0:
                node = max(flow_dict[node], key=flow_dict[node].get)
            else:
                break
                bProblem = True
        
        if bProblem == False:
            maxWalk.pop(0)
    else:
        flow_value = None
        maxWalk = None      
        
    return (flow_value, maxWalk)

def determineSourceSink(focalGraph,reverseFocalGraph,focalCoreDirected):
    
    if nx.is_directed_acyclic_graph (focalGraph):                
        lcd = UnitigGraph.getLowestCommonDescendant(focalGraph,focalCoreDirected)
            
        lca = UnitigGraph.getLowestCommonDescendant(reverseFocalGraph,focalCoreDirected)
                        
    else:
                
        print("Going to try and find lca/lcd on this local graph even though it is not a DAG")
                
        lcd = UnitigGraph.getLowestCommonDescendantG(focalGraph,focalCoreDirected)
            
        lca = UnitigGraph.getLowestCommonDescendantG(reverseFocalGraph,focalCoreDirected)

    return (lcd,lca)

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("gfa_file", help="assembly graph in gfa format")

    parser.add_argument("cog_file", help="unitig cog assignments")
    
    parser.add_argument("core_cogs", help="list of core cogs")
    
    parser.add_argument("kmer_length", help="kmer length assumed overlap")
    
    parser.add_argument("outFileStub", help="stub for output files")
    
    parser.add_argument('-c','--cov_file',nargs='?', help="unitig coverages")
    
    parser.add_argument('-e','--expansion_factor',nargs='?', help="expansion around core cogs",default=5.0,type=float)
    
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
    
    cogFragments = defaultdict(list)
    
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
        
            id = coreCog + "_" + str(g)
            bFragment = False
            bBubble = False
            nBubbleFound = 0
            nFocalFound = 0
            nReachableFound = 0
            nRSize = 0
            nBSize = 0
            
            focalGraph = nx.ego_graph(unitigGraph.directedUnitigBiGraph,corePlusName,radius=args.expansion_factor*cogLength,center=True,undirected=True,distance='weight')
            reverseFocalGraph = focalGraph.reverse()
            
            focalUnitigs = [n[:-1] for n in focalGraph.nodes()]
            focalCore = set(coreUnitigs).intersection(focalUnitigs)  
            focalCoreDirected = [] 
        
            for u in focalCore:
                focalCoreDirected.extend(getDirected(focalGraph,u))

            nFocalFound = len(focalCore)
            print("Graph\t" + str(g) + "\t" + str(nFocalFound))

            focalUGraph = unitigGraph.createUndirectedGraphSubset(focalUnitigs)
            
            focalUGraph.writeToGFA(args.outFileStub + id + "F.gfa")
        
            with open(args.outFileStub + id + "R.tsv",'w') as f:
                for x in focalCore:
                    f.write(x + "\n")
        
            (lcd, lca) = determineSourceSink(focalGraph,reverseFocalGraph,focalCoreDirected)
        
            if lcd is None or lca is None:
                
                print("Finding lca/lcd failed but these unitigs appear to be whole genes anyway...")
                l = 0
                for x in focalCore:
                    if focalUGraph.lengths[x] > 3.0*cogLength:
                        print(x)
                        focalXGraph = unitigGraph.createUndirectedGraphSubset([x])
                        focalXGraph.writeToGFA(args.outFileStub + id + "_" + str(l) + "X.gfa")
                        
                        if args.cov_file is not None:
                            focalXGraph.writeCovToCSV(args.outFileStub + id + "_" + str(l) + "X.csv")

                            l = l + 1
            else:
                bFragment = True
                
                reachable = get_labelled_subgraph(focalGraph, lca + lcd)
                reachable = add_connecting_nodes(reachable, focalGraph, lca + lcd)
                reachableU = [x[:-1] for x in reachable]
        
                reachableUGraph = unitigGraph.createUndirectedGraphSubset(reachableU)
    
                reachableGraph = focalGraph.subgraph(reachable)
                nRSize = len(reachableU)
                nReachableFound = len(set(reachableU) & focalCore)
                if nReachableFound == nFocalFound:
                    
                    reachableUGraph.writeToGFA(args.outFileStub + id + "R.gfa")
                    
                    if args.cov_file is not None:

                        (maxFlow,maxWalk) = getMaximumCoverageWalk(fullGraph,reachable,unitigGraph.covMap)
                        
                        maxFlows[id] = maxFlow
                        
                        print("Max flow: " + str(maxFlow))
                        
                        if maxWalk is not None:
                            contig = unitigGraph.getUnitigWalk(maxWalk)
            
                        covReachable = reachableUGraph.computeMeanCoverage(len(contig))
                        covs[id] = covReachable
                        seqs[id] = contig
                    
                    if args.cov_file is not None:
                            reachableUGraph.writeCovToCSV(args.outFileStub + id + "R.csv")
                else:
                    print("Incomplete Reachable\t" + str(g) + "\t" + str(nFocalFound - nReachableFound))
                    reachableUGraph.writeToGFA(args.outFileStub + id + "J.gfa")
        
    
                bubbleNodes = bubbleOut2(fullGraph, fullGraph, reverseFullGraph,lca[0],lcd[0],reachable)
                   
                coreFound = set([x[:-1] for x in bubbleNodes]) & focalCore
            
                nBubbleFound = len(coreFound)
            
                print("Bubble\t" + str(g) + "\t" + str(nBubbleFound))
                    
                bubbleU = [x[:-1] for x in bubbleNodes]
                nBSize = len(bubbleU)
                if len(bubbleU) > 0:
                    bBubble = True
                    bubbleUGraph = unitigGraph.createUndirectedGraphSubset(bubbleU)
                
                    if nBubbleFound == nFocalFound:
                        bubbleUGraph.writeToGFA(args.outFileStub + id + "B.gfa")
                    
                        if args.cov_file is not None:
                            bubbleUGraph.writeCovToCSV(args.outFileStub + id + "B.csv")
                    else:
                        print("Incomplete Bubble\t" + str(g) + "\t" + str(nFocalFound - nBubbleFound))
                        bubbleUGraph.writeToGFA(args.outFileStub + id + "I.gfa")
                    
                        if args.cov_file is not None:
                            bubbleUGraph.writeCovToCSV(args.outFileStub + id + "I.csv")
                else:
                    print("Empty Bubble\t" + str(g) + "\t" + str(len(bubbleU)))
            
            for x in focalCore:
                coreUnitigs.remove(x)
            
            cogFragments[coreCog].append((id,bFragment,bBubble,nRSize, nBSize, nFocalFound,nReachableFound,nBubbleFound))
            
            g = g + 1
            
    with open(args.outFileStub + 'fragments.tsv','w') as f:
        for coreCog,cogLength in coreCogs.items():
            fragmentList = cogFragments[coreCog]
        
            for fidx, fragment in enumerate(fragmentList):
                fList = list(fragment)
                fString = "\t".join([str(x) for x in fList])
            
                f.write(coreCog + "\t" + str(fidx) + "\t" + fString + "\n")
    
    
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
