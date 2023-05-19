import networkx as nx
from topology import *
from pcn import *
import time

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

def greedy_algorithm (Graph, node_improve, channels, alpha = 0.5, cycle = True):
    if node_improve not in Graph.nodes:
        Graph.add_node(node_improve)
    
    selected_node = []
    cc_after = []

    end = False
    while(len(selected_node) < channels and end == False):
        max_reward = 0
        max_node = None
        network_nodes = Graph.nodes()
        for node in tqdm(network_nodes, desc='Testing nodes to connect to'):
            if node == node_improve:
                continue
            if Graph.has_edge(node_improve, node) == True:
                continue

              
            
            Graph.add_edge(node_improve, node, fee_base_msat = 100, fee_proportional_millionths=50)
            Graph.add_edge(node, node_improve, fee_base_msat = 100, fee_proportional_millionths=50)
            
            payment_graph = make_graph_payment(Graph, 4104693)
            for (i,j) in payment_graph.edges():
                if type(payment_graph[i][j]['fee']) is str:
                    payment_graph[i][j]['fee'] = int(payment_graph[i][j]['fee'])
            new_cc = nx.closeness_centrality(payment_graph, u=node_improve, distance='fee')
            new_bc = nx.betweenness_centrality(payment_graph, normalized = True, weight='fee')
            
            new_reward = (alpha*new_bc[node_improve] + (1-alpha)*new_cc)
            

            if new_reward >= max_reward:
                if cycle == True:
                    if len(selected_node) != 0:
                        for node1 in selected_node:
                            if Graph.has_path(node,node1) == True or Graph.has_path(node1, node) == True:
                                max_reward = new_reward
                                max_node = node
                    else:
                        max_reward = new_reward
                        max_node = node
                else: 
                    max_reward = new_reward
                    max_node = node
            Graph.remove_edge(node_improve, node)
            Graph.remove_edge(node, node_improve)
        
        if len(cc_after) != 0:
            if max_reward < cc_after[-1]:
                end = True
                break
        Graph.add_edge(max_node, node_improve, fee_base_msat = 100, fee_proportional_millionths=50)
        Graph.add_edge(node_improve, max_node, fee_base_msat = 100, fee_proportional_millionths=50)
        cc_after.append(max_reward)
        selected_node.append(max_node)
    return selected_node, cc_after

def sample_pdf (pdf, k: int):
    """
    Samples k items from a PDF.
    """
    if k < 0 or k > len(pdf):
        raise ValueError('k must be between 0 and ' + str(len(pdf)))
    keys, v = zip(*list(pdf.items()))
    if k >= len(keys):
        return keys
    res = np.random.choice(keys, k, replace = False, p = v)
    return res

def manipulate_pdf (pdf, skew, smooth):
    """
    Skews or smooths a PDF. Skewing a distribution in done by squaring the probabilities and normalizing the pdf.
    Smoothing a distribution is done by mixing it with a uniform distribution.
    Although both functions should not be used at the same time, we don't prevent this.
    Adapted from https://github.com/renepickhardt/lightning-network-autopilot/blob/master/lib_autopilot.py 
    """
    if not skew and not smooth:
        return pdf

    length = len(pdf)

    if skew:
        pdf = {k:v**2 for k,v in pdf.items()}
        s = sum(pdf.values())
        pdf = {k:v/s for k,v in pdf.items()}

    if smooth:
        pdf = {k:0.5*v + 0.5/length for k,v in pdf.items()}
    
    return pdf


def get_uniform_distribution_pdf (Graph: nx.DiGraph):
    """
    Generates a uniform distribution of nodes in the network. Adapted from
    https://github.com/renepickhardt/lightning-network-autopilot/blob/master/lib_autopilot.py 
    """
    return {n:1/Graph.number_of_nodes() for n in Graph.nodes()}

def get_centrality_distribution_pdf (Graph: nx.DiGraph, skew = False, smooth = False):
    """
    Generates a distribution proportional to the betweenness centrality of nodes in the network. Adapted from
    https://github.com/renepickhardt/lightning-network-autopilot/blob/master/lib_autopilot.py 
    """
    pdf = {}
    cumsum = 0

    """We use fees to determine a node's BC to properly imitate a PCN"""
    payment_graph = make_graph_payment(Graph, 4104693)
    for n, score in nx.betweenness_centrality(payment_graph, weight='fee').items():
        pdf[n] = score
        cumsum += score
    
    pdf = {k:v/cumsum for k, v in pdf.items()}

    return manipulate_pdf(pdf, skew, smooth)

def get_degree_distribution_pdf (Graph: nx.DiGraph, skew = False, smooth = False):
    """
    Generates a distribution proportional to the betweenness centrality of nodes in the network. Adapted from
    https://github.com/renepickhardt/lightning-network-autopilot/blob/master/lib_autopilot.py 
    """
    pdf = {}
    cumsum = 0
    for n, score in dict(Graph.degree()).items():
        pdf[n] = score
        cumsum += score
    
    pdf = {k:v/cumsum for k, v in pdf.items()}

    return manipulate_pdf(pdf, skew, smooth)


def get_rich_nodes_pdf (Graph: nx.DiGraph, skew = False, smooth = False):
    """
    Generates a distribution proportional to the capacity of nodes in the network. Adapted from
    https://github.com/renepickhardt/lightning-network-autopilot/blob/master/lib_autopilot.py 
    """
    rich_nodes = {}
    network_capacity = 0

    for node in Graph.nodes():
        node_capacity = 0
        for neighbor in Graph.neighbors(node):
            node_capacity += int(Graph[node][neighbor]['capacity'])
        network_capacity += node_capacity
        rich_nodes[node] = node_capacity

    rich_nodes = {k:v/network_capacity for k, v in rich_nodes.items()}
    return manipulate_pdf(rich_nodes, skew, smooth)



def degree_only (Graph, node_improve, channels, parameter, alpha = 0.5, beta = 0.5, cycle = True):
    if node_improve not in Graph.nodes:
        Graph.add_node(node_improve)

    increment_shortest_path(Graph)
    
    new_edges = []
    selected_node = []
    cc_after = []

    if parameter == 'degree':
        centralized = get_k_most_centralized_nodes(Graph, 400)
    if parameter == 'bc':
        centralized = get_k_most_centralized_nodes_bc(Graph, 400)
    if parameter == 'cc':
        centralized = get_k_most_centralized_nodes_cc(Graph, 400)
    count = 1
    reward = 0
    while(len(new_edges) < channels):
        node = centralized[-count]
        count += 1
        if node == node_improve:
            continue
        if Graph.has_edge(node_improve, node) == True:
            continue
          
        
        Graph.add_edge(node_improve, node, fee_base_msat = 1001)
        new_cc = nx.closeness_centrality(Graph, u=node_improve,distance="fee_base_msat")
        new_bc = edges_betweenness_centrality(Graph, 25)
        
        if (node_improve, node) not in new_bc:
            reward = (alpha*new_bc[(node, node_improve)] + beta*new_cc)/2
        else:
            reward = (alpha*new_bc[(node_improve, node)] + beta*new_cc)/2

        if cycle == True:
            if len(selected_node) != 0:
                for node1 in selected_node:
                    if Graph.has_edge(node,node1) == True or Graph.has_edge(node1, node) == True:
                        cc_after.append(reward)
                        selected_node.append(node)
                        new_edges.append((node,node_improve))
                    else:
                        Graph.remove_edge(node_improve, node)
            else:
                cc_after.append(reward)
                selected_node.append(node)
                new_edges.append((node,node_improve))
        else: 
            cc_after.append(reward)
            selected_node.append(node)
            new_edges.append((node,node_improve))
        
    return selected_node, cc_after
