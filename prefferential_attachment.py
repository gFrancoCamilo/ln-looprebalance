import networkx as nx
from topology import *

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

def incremental_closeness (Graph, node_improve, channels):
    cc = nx.closeness_centrality(Graph)
    new_edges = []
    selected_node = []
    while(len(new_edges) < channels):
        max_reward = 0
        network_nodes = Graph.nodes()
        for node in network_nodes:
            if node == node_improve:
                continue
            if Graph.has_edge(node_improve, node) == True:
                continue
                 
            new_cc = nx.incremental_closeness_centrality(Graph, (node_improve, node), cc, True)
            new_reward = new_cc[node_improve]
            if new_reward >= max_reward:
                max_reward = new_reward
                max_node = node

        Graph.add_edge(max_node,node_improve)
        selected_node.append(max_node)
        new_edges.append((max_node,node_improve))
    return selected_node

def incremental_betweenness (Graph, node_improve, channels):
    bc = nx.edge_betweenness_centrality(Graph)
    new_edges = []
    selected_node = []
    while(len(new_edges) < channels):
        max_reward = 0
        network_nodes = Graph.nodes()
        for node in network_nodes:
            if node == node_improve:
                continue
            if Graph.has_edge(node_improve, node) == True:
                continue
            
            Graph.add_edge(node_improve, node)
            new_bc = nx.edge_betweenness_centrality(Graph)
            
            if node > node_improve:
                new_reward = new_bc[(node_improve, node)]
            else:
                new_reward = new_bc[(node, node_improve)]
            if new_reward >= max_reward:
                max_reward = new_reward
                max_node = node
            Graph.remove_edge(node_improve, node)

        Graph.add_edge(max_node,node_improve)
        selected_node.append(max_node)
        new_edges.append((max_node,node_improve))
    return selected_node
