import networkx as nx
import datetime
import itertools
import queue
import numpy as np
from multiprocessing import Pool


def generate_timestamps ():
    """
    The Lightning network graph-files are named by the unix-time of when the snapshot was taken.
    Each snapchot was taken with a difference of two weeks. This function returns a list with 
    the names of every file available in the directory 'graph'
    """
    filenames = []
    dates = []
    """1579575600 is the timestamp of the first snapshot"""
    timestamp_iterator = 1579575600
    """
    1630378800 is the timestamp of the last snapshot. The loop iterates through every two
    weeks between the first and last timestamp and creates a list of the unix-time of
    every snapshot. It also creates a human-readable date of the timestamp. As an example,
    the first timestamp (1579575600) is added to the dates list as 21 Jan 2018
    """
    while timestamp_iterator <= 1630378800:
        filenames.append('ln-snapshots/graph-' + str(timestamp_iterator))
        dates.append(datetime.datetime.fromtimestamp(timestamp_iterator).strftime('%d-%m-%Y'))
        timestamp_iterator = timestamp_iterator + 2*7*24*3600
    return filenames, dates

def graph_names (option: str = 'jul 2022'):
    """
    graph_names reads Lightning snapshots in gml format and returns a NetworkX graph.
    The function offers three snapshots option from three different time-periods. The
    first one is from July 2021, the second is from Jan 2022, and the third is from
    July 2022. We cnsider that the snapshots are directed graphs with channel capacity
    attributes already present. Furthermore, the function returns the highest connected
    component of the graph.
    """
    filenames = ['./ln-snapshots/ln-topology-20210701-recovered-capacitated.gml',
                './ln-snapshots/ln-topology-20220101-recovered-capacitated.gml',
                './ln-snapshots/ln-topology-20220701-recovered-capacitated.gml']
    if option == "jul 2021":
        Graph = create_graph(filenames[0], 'gml')
    if option == "jan 2022":
        Graph = create_graph(filenames[1], 'gml')
    if option == "jul 2022":
        Graph = create_graph(filenames[2], 'gml')

    """Removing small graph-components"""
    graph_components = sorted(nx.strongly_connected_components(Graph), key=len, reverse=True)
    Graph = Graph.subgraph(graph_components[0])
    return Graph

def create_graph (filename: str, format: str):
    """
    Reads file that contains the Lightning network graph encoded in graphml format and
    returns a NetworkX undirected graph
    """
    try:
        if format == "gml":
            Graph = nx.read_gml(filename)
        else:
            Graph = nx.read_graphml(filename)
            Graph = Graph.to_undirected()
        return Graph
    except:
        raise Exception ('Invalid graph filename or format')

def count_node_triangles (Graph, nodes = None):
    """
    count_node_triangles counts the number of triangles a node is part of in the network.
    It receives the network Graph and the node to evaluate the number of triangles.
    """
    try:
        return nx.triangles(Graph, nodes)
    except:
        raise Exception ('Invalid node or graph')

def chunks(l, n):
    """
    Divide a list of nodes `l` in `n` chunks
    """
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x

def edges_betweenness_centrality (Graph, processes = None):
    """
    Calculates the betweenness centrality of every node in a parallel way. As the Lightning
    network is composed of thousands of nodes, using multiprocessing can help to accelerate
    some calculations. The betweenness centrality is key to develop a prefferential 
    attachment strategy that is profitable for the user
    """
    try:
        p = Pool(processes=processes)
        node_divisor = len(p._pool) * 4
        node_chunks = list(chunks(Graph.nodes(), Graph.order() // node_divisor))
        num_chunks = len(node_chunks)
        bt_sc = p.starmap(
            nx.edge_betweenness_centrality_subset,
            zip(
                [Graph] * num_chunks,
                node_chunks,
                [list(Graph)] * num_chunks,
                [True] * num_chunks,
                ["fee_base_msat"] * num_chunks,
            ),
        )

        """Reduce the partial solutions"""
        bt_c = bt_sc[0]
        for bt in bt_sc[1:]:
            for n in bt:
                bt_c[n] += bt[n]
        return bt_c
    
    except:
        raise Exception ('Invalid Graph. Could not calculate betweenness centrality')

def nodes_closeness_centrality (Graph):
    """
    Calculates the closenness centrality of every node. The closenness centrality is key
    to develop an algorithm that makes issuing transaction to other nodes cheap.
    """
    try:
        return nx.closeness_centrality(Graph)
    except:
        raise Exception ('Invalid Graph. Could not calculate closenness centrality')

def get_node_diff (Graph2, Graph1):
    """
    Returns the set of nodes present in Graph2 that are not present in Graph1. Thus,
    the function is useful to find the nodes that were introduced in a period between
    two dates
    """
    try:
        return Graph2.nodes() - Graph1.nodes()
    except:
        raise Exception ('Could not diff the two graphs')

def triadic_census (Graph):
    """
    Returns the triadic census of the graph. As the native NetworkX function does not 
    provide a triadic census for undirected graphs, we iterate through combinations of
    three nodes in the graph to create our own triadic census in undirected graphs.
    Solution inspirated by: https://stackoverflow.com/questions/54730863/how-to-get-
    triad-census-in-undirected-graph-using-networkx-in-python
    """
    try:
        triadic_class = {}
        for nodes in itertools.combinations(Graph.nodes, 3):
            n_edges = Graph.subgraph(nodes).number_of_edges()
            triadic_class.setdefault(n_edges, []).append(nodes)
        del triadic_class[0]
        return triadic_class
    except:
        raise Exception ('Invalid Graph. Could not compute the triadic census')

def get_k_most_centralized_nodes (Graph, k):
    """
    get_k_most_centralized_nodes returns the k nodes with highest degree in the
    network.
    """
    centralization = nx.degree_centrality(Graph)
    centralization = dict(sorted(centralization.items(), key=lambda item: item[1]))
    centralization_list = list(centralization.keys())[-k:]
    return centralization_list

def get_k_most_centralized_nodes_bc (Graph, k):
    """
    get_k_most_centralized_nodes returns the k nodes with highest betweenness centrality
    in the network.
    """
    centralization = nx.betweenness_centrality(Graph)
    centralization = dict(sorted(centralization.items(), key=lambda item: item[1]))
    centralization_list = list(centralization.keys())[-k:]
    return centralization_list

def get_k_most_centralized_nodes_cc (Graph, k):
    """
    get_k_most_centralized_nodes returns the k nodes with highest closeness centrality
    in the network.
    """
    centralization = nx.closeness_centrality(Graph)
    centralization = dict(sorted(centralization.items(), key=lambda item: item[1]))
    centralization_list = list(centralization.keys())[-k:]
    return centralization_list

def get_k_most_centralized_edges_bc (Graph: nx.DiGraph, k: int):
    """
    get_k_most_centralized_edges_bc returns the k edges with highest betweenness centrality
    in the network.
    """
    centralization = nx.edge_betweenness_centrality(Graph)
    centralization = dict(sorted(centralization.items(), key=lambda item: item[1]))
    centralization_list = list(centralization.keys())[-k:]
    return centralization_list


def increment_shortest_path (Graph):
    """
    increment_shortest_path increments the attribute fee_base_msat in 1.
    """
    dic = nx.get_edge_attributes(Graph, "fee_base_msat")
    for edge in dic:
        (u, v) = edge
        Graph[u][v]["fee_base_msat"] = dic[edge] + 1

def snowball_sample (Graph, node = None, size = 200):
    """
    snowball_sample uses snowball sampling technique to sample the original Lightning
    graph. It receives the original graph, a node to kick off the algorithm and the
    maximum size of the sampled graph. If no node is passed as argument, it selects the
    highest degree node to start the algorithm. If random is passed, it randomly selects
    a node from the graph.
    """
    copy = Graph.copy()
    if node == 'random':
        node = np.random.choice(copy.nodes())
    if node == None:
        node = max(copy.nodes(), key=lambda node: Graph.degree(node))
    if Graph.number_of_nodes() < size:
        return list()
    q = queue.Queue()
    q.put(node)
    subgraph = list()
    subgraph.append(node)
    while not q.empty():
        for node in copy.neighbors(q.get()):
            if len(subgraph) < size:
                q.put(node)
                subgraph.append(node)
            else:
                break     
    remaining_nodes = []       
    for node in copy.nodes():
        if node not in subgraph:
            remaining_nodes.append(node)
    copy.remove_nodes_from(remaining_nodes)
    return copy