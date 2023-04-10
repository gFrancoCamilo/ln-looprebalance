from topology import *
import networkx as nx
import random
import copy
from tqdm import tqdm

def validate_graph (Graph):
    """
    validate_graph checks if for every (i,j) edge in the network
    graph, another (j,i) edge exists. If this rule is broken, the
    (i,j) edge is removed from the graph.
    """
    edges_to_remove = []
    desc = 'Validating Graph'
    for (i,j) in tqdm(Graph.edges, desc=desc):
        if i not in Graph.neighbors(j):
            edges_to_remove.append((i,j))
    for (i,j) in edges_to_remove:
        Graph.remove_edge(i,j)
    return Graph

def set_capacities (Graph, option: str):
    if option == 'lightning':
        capacities = get_lightning_capacities()
        for (i,j) in tqdm(Graph.edges, desc='Assigning capacities'):
            if 'capacity' not in Graph[i][j]:
                Graph[i][j]['capacity'] = random.choice(capacities)
                Graph[j][i]['capacity'] = Graph[i][j]['capacity']
    else:
        raise Exception ('Invalid option for capacity setting')
    return Graph

def get_lightning_capacities ():
    Graph = graph_names ('jul 2022')
    Graph = validate_graph(Graph)

    capacities = []
    for (i,j) in Graph.edges():
        capacities.append(Graph[i][j]['capacity'])
    
    return capacities


def set_balance (Graph, option: str = '99-1'):
    """
    set_balance initiates the balance for every channel on the network.
    We define channel capacity as the funds collectively locked by
    both parties in the channel. On the other hand, balance is defines
    how the channel capacity is splitted between both nodes in a channel.
    Thus, the capacity c_ij of a channeel (i,j) is the sum b(i)+b(j) of the
    balances of each node. The function presents two option: half and 99-1.
    If half is selected, the balance of every channel is splitted in half,
    i.e., if a channel (i,j) has capacity 10, each node is assigned balance
    b(i) = b(j) = 5. If the option 99-1 is selected, the algorithm sorts one
    of the nodes and assigns them 99% of the channel capacity and 1% to the
    other party.
    """
    desc = 'Setting balance with option ' + str(option)
    for (i,j) in tqdm(Graph.edges, desc=desc):
        if 'balance' not in Graph[i][j]:
            capacity = int(Graph[i][j]['capacity'])
            if option == 'half':
                Graph[i][j]['balance'] = capacity//2
                Graph[i][j]['payments'] = []
                Graph[j][i]['balance'] = capacity - Graph[i][j]['balance']
                Graph[j][i]['payments'] = []
            elif option == '99-1':
                coin = random.randint(0,1)
                if coin == 0:
                    Graph[i][j]['balance'] = round(0.99*capacity)
                    Graph[i][j]['payments'] = []
                    Graph[j][i]['balance'] = capacity - Graph[i][j]['balance']
                    Graph[j][i]['payments'] = []
                else:
                    Graph[j][i]['balance'] = round(0.99*capacity)
                    Graph[i][j]['payments'] = []
                    Graph[i][j]['balance'] = capacity - Graph[j][i]['balance']
                    Graph[j][i]['payments'] = []
            else:
                raise Exception ('No valid option to set balance selected')
    return Graph

def set_balance_ln (Graph, alpha: float = 0.01):
    """
    set balance_ln initializes the network balance simulating
    a real-world scenario. Basically, central channels, i.e.,
    channels between highest degree nodes, are kept balanced
    half-half. Meanwhile, the rest of the nodes is assigned a
    balance of 99-1 in its channels.
    """
    number_nodes = round(Graph.number_of_nodes()*alpha)
    k_central_nodes = get_k_most_centralized_nodes (Graph, number_nodes)
    k_central_nodes_dict = dict.fromkeys(k_central_nodes, "True")

    desc = 'Setting balance in the most central nodes'
    for node in tqdm(k_central_nodes, desc=desc):
        for neighbor in Graph.neighbors(node):
            if neighbor in k_central_nodes_dict:
                Graph[node][neighbor]['balance'] = int(Graph[node][neighbor]['capacity'])//2
                Graph[node][neighbor]['payments'] = []
                Graph[neighbor][node]['balance'] = int(Graph[node][neighbor]['capacity']) - Graph[node][neighbor]['balance'] 
                Graph[neighbor][node]['payments'] = []
            else:
                coin = random.randint(0,1)
                if coin == 0:
                    Graph[node][neighbor]['balance'] = round(0.99*int(Graph[node][neighbor]['capacity']))
                    Graph[node][neighbor]['payments'] = []
                    Graph[neighbor][node]['balance'] = int(Graph[node][neighbor]['capacity']) - Graph[node][neighbor]['balance']
                    Graph[neighbor][node]['payments'] = []
                else:
                    Graph[neighbor][node]['balance'] = round(0.99*int(Graph[node][neighbor]['capacity']))
                    Graph[neighbor][node]['payments'] = []
                    Graph[node][neighbor]['balance'] = int(Graph[node][neighbor]['capacity']) - Graph[neighbor][node]['balance']
                    Graph[node][neighbor]['payments'] = []
    
    desc = 'Setting balance in the rest of the network'
    for (i,j) in tqdm(Graph.edges(), desc=desc):
        if 'balance' not in Graph[i][j]:
            capacity = int(Graph[i][j]['capacity'])
            coin = random.randint(0,1)
            if coin == 0:
                Graph[i][j]['balance'] = round(0.99*capacity)
                Graph[i][j]['payments'] = []
                Graph[j][i]['balance'] = capacity - Graph[i][j]['balance']
                Graph[j][i]['payments'] = []
            else:
                Graph[j][i]['balance'] = round(0.99*capacity)
                Graph[j][i]['payments'] = []
                Graph[i][j]['balance'] = capacity - Graph[j][i]['balance']
                Graph[i][j]['payments'] = []
    return Graph

def find_shortest_path (Graph, s, t):
    """
    find_shortest_path returns the shortest path between a source and a
    destination. Following BOLT #7 (https://github.com/lightning/bolts/blob/master/07-routing-gossip.md#htlc-fees),
    we use Djikstra algorithm with weight equal fb + v*fr, where fb is
    the fixed base fee of an edge, v is the value of the payment in
    satoshis and fr is the fee rate of the channel.
    """
    return nx.shortest_path(Graph, source = s, target = t, weight = 'fee')

def make_graph_payment (Graph: nx.DiGraph, value: int) -> nx.DiGraph:
    """
    make_graph_payment transforms the regular network graph into a payment graph.
    As payment fees vary in relation to the payment values, we set a new attribute
    to the graph that provides the fee amount that a user will have to pay to route
    the payment through that channel. This payment graph will then be used to compute
    the shortest path that the user can use.
    """
    Graph_copy = Graph.copy()

    for (i,j) in Graph.edges:
        fee_base = int(Graph[i][j]['fee_base_msat'])
        fee_rate = int(Graph[i][j]['fee_proportional_millionths'])
        Graph_copy[i][j]['fee'] = round(fee_base + value*(fee_rate/1000000))
    
    return Graph_copy

def make_payment (Graph, s, t, value, path = None, debug = False):
    """
    make_payment attemps to issue a payment of a certain value from
    a source node to a destination node. The function implements the
    PCN routing-logic where each payment reduces the capacity of a
    channel of forwarding future payments. 
    """

    Graph_copy = make_graph_payment(Graph, value)
    
    if path == None:
        hops = find_shortest_path(Graph_copy, s, t)
    else:
        hops = path
    
    """After finding the shortest path, the graph_copy is useless to us as we change the original Graph"""
    del(Graph_copy)

    """Check if there is a path to route"""
    if len(hops) == 0:
        raise Exception ('Path is empty')

    index = 0
    if debug == True:
        print("Hops: " + str(hops))
    
    while index < (len(hops) - 1):
        index += 1
        if value > Graph[hops[index-1]][hops[index]]['balance']:
            if debug == True:
                print ("Payment value: " + str(value))
                print ("Channel balance: " + str(Graph[hops[index-1]][hops[index]]['balance']))
            raise Exception ('Not enough balance on path')

    index = 0
    while index < (len(hops) - 1):
        index += 1
        try:
            if Graph[hops[index-1]][hops[index]]['balance'] - value < 0:
                raise Exception ('Out of funds')
            else:
                Graph[hops[index-1]][hops[index]]['balance'] = Graph[hops[index-1]][hops[index]]['balance'] - value
                Graph[hops[index]][hops[index-1]]['balance'] = Graph[hops[index]][hops[index-1]]['balance'] + value

                Graph[hops[index-1]][hops[index]]['payments'].append(value*(-1))
                Graph[hops[index]][hops[index-1]]['payments'].append(value)
        except:
            raise Exception ('Could not issue payment')

def make_payment_lnd (Graph: nx.DiGraph, source, target, value: int, debug: bool = False):
    """copy holds a copy of the Graph so the original is not modified"""
    graph_copy = make_graph_payment(Graph, value)
    
    """If there is no path from source to destination, it is useless to attempt a payment"""
    if nx.has_path (graph_copy, source, target) == False:
        raise Exception ('No path between source and destination available')
    
    """If source node attempts to issue a payment higher than its local balance, it will fail"""
    if value > get_node_balance(graph_copy, source):
        raise Exception ('Local balance is insufficient to make payment')
    
    paths_tried = 0

    while (nx.has_path(graph_copy,source,target) and paths_tried < 50):
        """index will iterate through the hops"""
        index = 0

        """try_another_path will indicate if the source needs to find another path to destination"""
        try_another_path = False

        hops = find_shortest_path(graph_copy, source, target)

        """If a channel can not be used to route the payment, we remove the edge and retry the Dijkstra algorithm"""
        while index < (len(hops) - 1):
            index += 1
            if value > graph_copy[hops[index-1]][hops[index]]['balance']:
                graph_copy.remove_edge(hops[index-1],hops[index])
                try_another_path = True
                break
        
        paths_tried += 1

        if try_another_path == False:
            break
    
    if try_another_path == True:
        raise Exception ('No path found')
    
    index = 0
    while index < (len(hops) - 1):
        index += 1
        try:
            if Graph[hops[index-1]][hops[index]]['balance'] - value < 0:
                raise Exception ('Out of funds')
            else:
                Graph[hops[index-1]][hops[index]]['balance'] = Graph[hops[index-1]][hops[index]]['balance'] - value
                Graph[hops[index]][hops[index-1]]['balance'] = Graph[hops[index]][hops[index-1]]['balance'] + value

                Graph[hops[index-1]][hops[index]]['payments'].append(value*(-1))
                Graph[hops[index]][hops[index-1]]['payments'].append(value)

        except:
            raise Exception ('Could not issue payment')



def get_node_balance (Graph, node, debug = False):
    """
    get_node balance computes the balance of a given node in the network.
    The node balance nb_i of a node i is defined as the sum of the balance
    of each channel i participates. As an example, if i has 3 channel (i,j),
    (i,k), and (i,l), with nodes j, k, and l, respectivaly, and has a balance
    b_1(i), b_2(i), b_3(i) in each channel, the node balance of i nb_i = 
    b_1(i) + b_2(i) + b_3(i). Illustrating

    ┌─────┐
    │     │
    │  L  │     Each arrow represents a channel.
    │     │     Each square represents a node.
    └┬────┘     The number over each channel represents the balance of the channel.
    2│  ▲       nb_I = 7 + 3 + 4 = 14
     │  │
     ▼  │7
    ┌───┴─┐ 3     ┌─────┐
    │     ├──────►│     │
    │  I  │       │  J  │
    │     │◄──────┤     │
    └┬────┘     5 └─────┘
    4│  ▲
     │  │
     ▼  │1
    ┌───┴─┐
    │     │
    │  K  │
    │     │
    └─────┘
    """
    try:
        neighbors = [n for n in Graph.neighbors(node)]
        node_balance = 0
        for neighbor in neighbors:
            node_balance += Graph[node][neighbor]['balance']
        if debug == True:
            print("\nNode balance " + str(node) + " : " + str(node_balance))
        return node_balance
    except:
        raise Exception("Could not compute node balance")

       