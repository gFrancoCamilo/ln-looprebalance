import networkx as nx
from topology import all

def triadic_closure (Graph, node, number_channels, triadic_census, alpha, beta):
    """
    Implements a connection establishment heuristic for already existing nodes
    based on the concept of triadic closure. This function receives a node in
    the network and the number of channels this node wants to create.
    The function aims at creating new connections while respecting two major
    goals: 1) extending the channel lifecycle and 2) being more profitable.
    To achieve 1), we priviledge 3-node creating 3-node cycles, which can be
    used by rebalancing algorithms to keep the channel balanced for the
    highest possible amount of time. To achieve 2), we evaluate new connections
    through their betweenness and closeness centrality measure to verify if
    the new connection make the node more profitable or not.
    """