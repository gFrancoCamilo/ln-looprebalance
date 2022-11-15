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

def betweenness_improvement (Graph, node_improve, channels):
    new_edges = []
    while(len(new_edges) < channels):
        max_reward = 0
        selected_node = []
        network_nodes = Graph.nodes()
        for node in network_nodes:
            if node == node_improve:
                continue
            if Graph.has_edge(node_improve, node) == True:
                continue

            Graph.add_edge(node_improve, node)
            bc = nodes_betweenness_centrality(Graph)
            new_reward = bc[node_improve]
            if new_reward >= max_reward:
                max_reward = new_reward
                max_node = node
            else:
                Graph.remove_edge(node_improve, node)
