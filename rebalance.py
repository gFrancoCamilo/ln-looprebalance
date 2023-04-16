import threading
import networkx as nx
import time
import tqdm
from cycle_finder import *
from pcn import *

def init_rebalance (Graph: nx.DiGraph, event: threading.Event, channel: tuple, threshold: float = 0.3, delay: int = 5):
    """
    init_rebalance starts the thread that monitor channels for rebalancing
    """
    thread = threading.Thread(target = monitor_channel, args=(Graph, event, channel, threshold, delay,))
    thread.start()

def monitor_channel (Graph: nx.DiGraph, event: threading.Event, channel: tuple, threshold: float = 0.3, delay: int = 5):
    """
    monitor_channel constantly checks if a channel is imbalanced and rebalances it if it.
    The imbalance measurement is done by checking what percentage of capacity each user holds.
    If that percentage falls below a threshold, the channel must be rebalanced. Thus, the function
    receives as arguments the network Graph, an event, that is used to stop the thread, the channel
    being monitored, the threshold and the delay, which defines the interval that the function
    checks the channel balance.
    """
    while True:
        if check_rebalance (Graph, channel, threshold) == True:
            rebalance_half(Graph, channel)

        time.sleep(delay)

        """If event is set, we can exit the loop because the simulation is finished"""
        if event.is_set():
            break

def check_rebalance (Graph: nx.DiGraph, channel: tuple, threshold: float = 0.3) -> bool:
    """
    check_rebalances check if a given channel balance is below a certain threshold or
    not. It return true if the channel balance is below the provided threshold and False
    otherwise.
    """
    (i,j) = channel
    if Graph[i][j]['balance']/int(Graph[i][j]['capacity']) < threshold:
        return True
    return False

def rebalance_half (Graph: nx.DiGraph, channel: tuple):
    """
    rebalance_half gets an imbalanced channel and issues a payment to balance it.
    This particular function aims at rebalancing the channel, so the capacity is equally
    splitted among the participants, i.e., the capacity is splittd in half. As an example,
    suppose we have a channel between nodes n1 and n2 with capacity c(n1,n2) = 10. At a moment,
    the channel is imbalanced with n1's balance b(n1) = 1 and n2's balance b(n2) = 9. User n1 can
    rebalance this channel by issuing a circular self-payment of value equal to 4. Thus, after the
    payment is complete, each node has exactly half of the capacity as its balance. 
    """
    (i,j) = channel 
    
    value = int(Graph[i][j]['capacity'])/2 - Graph[i][j]['balance']

    """As our rebalance depends on self-payments, we check possible paths we can issue the rebalance payment"""
    cycles = find_cycle(Graph, channel, i, value)
    
    """If no cycle is found, it is not possible to rebalance the channel"""
    if len(cycles) == 0:
        raise Exception ('Impossible to rebalance')
    
    """For each cycle returned, we attempt to rebalance the channel through that path"""
    rebalanced = False
    for cycle in tqdm(cycles, desc='Trying to rebalance'):
        if rebalanced == True:
            break
        try:
            make_payment(Graph, i, i, value, cycle)
            rebalanced = True
        except Exception as e:
            continue
    
    """If rebalanced is False, it means that the rebalance payment couldn't be routed through the cycles"""
    if rebalanced == False:
        raise Exception ('Could not rebalance channel')

def pickhardt_and_nowostawski (Graph: nx.DiGraph, node: str):
    """
    pickhardt_and_nowostawski implements the rebalancing algorithm proposed by Pickhardt and Nowostawski
    published at ICBC (available at https://ieeexplore.ieee.org/document/9169456).
    """

    """Computing node balance coefficient"""
    node_balance = get_node_balance(Graph, node)
    node_capacity = 0
    for neighbor in Graph.neighbors(node):
        node_capacity += Graph[node][neighbor]['capacity']
    node_balance_coefficient = node_balance/node_capacity

    """Computing channel balance coefficients"""
    channel_balance_coefficients = [(neighbor, Graph[node][neighbor]['balance']/Graph[node][neighbor]['capacity']) for neighbor in Graph.neighbors(node)]

    """Checking which channel balance coefficients are higher than node coefficient"""
    imbalanced = []
    for (neighbor,channel_coefficient) in channel_balance_coefficients:
        if channel_coefficient - node_balance_coefficient > 0:
            imbalanced.append((neighbor, channel_coefficient))
            