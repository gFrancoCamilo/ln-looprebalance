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

def incremental_closeness (Graph, node_improve, channels, alpha = 0.5, beta = 0.5, cycle = True):
    if node_improve not in Graph.nodes:
        Graph.add_node(node_improve)
    
    cc = nx.closeness_centrality(Graph)
    bc = nx.edge_betweenness_centrality(Graph)
    new_edges = []
    selected_node = []
    cc_after = []
    while(len(new_edges) < channels):
        max_reward = 0
        network_nodes = Graph.nodes()
        for node in network_nodes:
            if node == node_improve:
                continue
            if Graph.has_edge(node_improve, node) == True:
                continue
                 
            new_cc = nx.incremental_closeness_centrality(Graph, (node_improve, node), cc, True)
            Graph.add_edge(node_improve, node)
            new_bc = nx.edge_betweenness_centrality(Graph)
            
            if (node_improve, node) not in new_bc:
                new_reward = (alpha*new_bc[(node, node_improve)] + beta*new_cc[node_improve])/2
            else:
                new_reward = (alpha*new_bc[(node_improve, node)] + beta*[node_improve])/2
            
            if new_reward >= max_reward:
                if cycle == True:
                    if len(selected_node) != 0:
                        for node1 in selected_node:
                            if Graph.has_edge(node,node1) == True or Graph.has_edge(node1, node) == True:
                                max_reward = new_reward
                                max_node = node
                    else:
                        max_reward = new_reward
                        max_node = node
                else: 
                    max_reward = new_reward
                    max_node = node
            Graph.remove_edge(node_improve, node)
        Graph.add_edge(max_node,node_improve)
        cc_after.append(max_reward)
        selected_node.append(max_node)
        new_edges.append((max_node,node_improve))
    return selected_node, cc_after
