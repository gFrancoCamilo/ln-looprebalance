import networkx as nx

def find_cycle (Graph, channel, node):
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
        (i,j) = channel
        copy.remove_edge(i,j)
        if node == i:
            destination = j
        else:
            destination = i
        cycles = []
        neighbors = [n for n in copy[node]]
        for neighbor in neighbors:
            cycles.append(nx.shortest_path(copy, neighbor, destination, weight = 'fee_proportional_millionths'))
        for cycle in cycles:
            cycle.insert(0, node)
            cycle.append(node)
        return cycles
    except:
        raise Exception ("Could not find cycles")
