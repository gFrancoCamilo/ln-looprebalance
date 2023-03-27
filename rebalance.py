import threading
import networkx as nx
import time
import tqdm
from cycle_finder import *

def init_rebalance (Graph: nx.DiGraph, event: threading.Event, channel: tuple, threshold: float = 0.3, delay: int = 5):
    thread = threading.Thread(target = monitor_channel, args=(Graph, event, channel, threshold, delay,))
    thread.start()

def monitor_channel (Graph: nx.DiGraph, event: threading.Event, channel: tuple, threshold: float = 0.3, delay: int = 5):
    """
    monitor_channel 
    """
    while True:
        if check_rebalance (Graph, channel, threshold) == True:
            rebalance_half(Graph, channel)

        time.sleep(delay)

        """If event is set, we can exit the loop because the simulation is finished"""
        if event.is_set():
            break

def check_rebalance (Graph: nx.DiGraph, channel: tuple, threshold: float = 0.3) -> bool:
    (i,j) = channel
    if Graph[i][j]['balance']/int(Graph[i][j]['capacity']) < threshold:
        return True
    return False

def rebalance_half (Graph: nx.DiGraph, channel: tuple):
    (i,j) = channel 

    print('Rebalancing channel ' + str(channel))
    print('Before rebalancing: ' + str(Graph[i][j]['balance']))
    
    value = int(Graph[i][j]['capacity'])/2 - Graph[i][j]['balance']

    cycles = find_cycle(Graph, channel, i, value)
    
    if len(cycles) == 0:
        raise Exception ('Impossible to rebalance')
    
    rebalanced = False
    for cycle in tqdm(cycles, desc='Trying to rebalance'):
        if rebalanced == True:
            break
        try:
            make_payment(Graph, i, i, value, cycle)
            rebalanced = True
        except Exception as e:
            continue
    
    print('After rebalancing: ' + str(Graph[i][j]['balance']))
    if rebalanced == False:
        raise Exception ('Could not rebalance channel')
