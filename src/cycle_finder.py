import networkx as nx
from pcn import *

def find_cycle (Graph, channel, node, value=500, length=False):
    """
    find_cycle finds a cycle that a given node is inserted in.
    The function receives a node that in and the channel that is
    involved in cycles. To compute the cycles, the function removes
    the edge and calculates the shortest path from the node's neighbors
    to the neighbor that shared the removed edge with the node. Thus, we
    find multiple cycles that can be used for rebalancing in the network
    """
    try:
        """
        As we don't want to modify the regular Graph, we make a copy of it.
        Thus, we can remove the given channel without losing its attributes.
        """
        copy = Graph.copy()
        copy = make_graph_payment(Graph, value)
        (i,j) = channel
        copy.remove_edge(i,j)
        if node == i:
            destination = j
        else:
            destination = i
        cycles = []
        neighbors = [n for n in copy[node]]
        for neighbor in neighbors:
            if length == False:
                cycles.append(find_shortest_path(copy, neighbor, destination))
            else:
                cycles.append(find_shortest_path_cost(copy, neighbor, destination))
        if length == True:
            cost_cycle = []
            for (cost, path) in cycles:
                cost_cycle.append(cost)
            return min(cost_cycle)

            return min(cycles)
        for cycle in cycles:
            cycle.insert(0, node)
            cycle.append(node)
        if len(cycles) > 100:
            return cycles[:100]
        return cycles
    except:
        raise Exception ("Could not find cycles")
