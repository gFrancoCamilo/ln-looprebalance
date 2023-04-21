import networkx as nx
import time
import threading
from topology import *

"""
Defines a TimeWindow class that is used to compute the throughput that
passes through a node in the network. The idea is to use this information
to perform a demand-based rebalance.
"""
def init_window(Graph, channel: tuple, event: threading.Event, size: int = 5) -> None:
    """Runs the window routine as a thread"""
    (i,j) = channel
    thread = threading.Thread(target = window_routine, args=(event, Graph, channel, size,))
    thread.start()
    
def window_routine (event: threading.Event, Graph: nx.DiGraph, channel, size):
    last_analyzed = 0
    while True:
        """Check if payment arrived"""
        throughput, last_analyzed = compute_throughput(Graph, channel, size, last_analyzed)

        """Check again next window"""
        time.sleep(size)

        """Check for stop"""
        if event.is_set():
            (i,j) = channel
            break

"""Computes throughput of the node being analyzed"""
def compute_throughput (Graph: nx.DiGraph, channel, size: int, last_analyzed) -> int:
    (i,j) = channel
    start = last_analyzed
    if len(Graph[i][j]['payments']) == 0:
        return 0, start

    accumulator = 0
    while start < len(Graph[i][j]['payments']):
        accumulator += Graph[i][j]['payments'][start]
        start+=1

    return round(accumulator/size), start