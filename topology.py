import networkx as nx
import datetime
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
        filenames.append('graph-' + str(timestamp_iterator))
        dates.append(datetime.datetime.fromtimestamp(timestamp_iterator).strftime('%d-%m-%Y'))
        timestamp_iterator = timestamp_iterator + 2*7*24*3600
    return filenames, dates


def create_graph (filename: str):
    """
    Reads file that contains the Lightning network graph encoded in graphml format and
    returns a NetworkX graph
    """
    try:
        Graph = nx.read_graphml(filename)
        return Graph
    except:
        raise Exception ('Invalid graph filename or format')

def count_node_triangles (Graph, nodes = None):
    try:
        return nx.triangles(Graph, nodes)
    except:
        raise Exception ('Invalid node or graph')

def nodes_betweenness_centrality (Graph, processess = None):
    """
    Calculates the betweenness centrality of every node in a parallel way. As the Lightning
    network is composed of thousands of nodes, using multiprocessing can help to accelerate
    some calculations
    """
    try:
        p = Pool(processes=processes)
        node_divisor = len(p._pool) * 4
        node_chunks = list(chunks(G.nodes(), G.order() // node_divisor))
        num_chunks = len(node_chunks)
        bt_sc = p.starmap(
            nx.betweenness_centrality_subset,
            zip(
                [G] * num_chunks,
                node_chunks,
                [list(G)] * num_chunks,
                [True] * num_chunks,
                [None] * num_chunks,
            ),
        )

        # Reduce the partial solutions
        bt_c = bt_sc[0]
        for bt in bt_sc[1:]:
            for n in bt:
                bt_c[n] += bt[n]
        return bt_c
    
    except:
        raise Exception ('Invalid Graph. Could not calculate betweenness centrality')


    