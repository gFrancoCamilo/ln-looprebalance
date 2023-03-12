from topology import *
import networkx as nx
import random

def validate_graph (Graph):
    edges_to_remove = []
    for (i,j) in Graph.edges:
        if i not in Graph.neighbors(j):
            edges_to_remove.append((i,j))
    for (i,j) in edges_to_remove:
        Graph.remove_edge(i,j)
    return Graph

def set_balance (Graph, option: str = '99-1'):
    for (i,j) in Graph.edges:
        if 'balance' not in Graph[i][j]:
            capacity = int(Graph[i][j]['capacity'])
            if option == 'half':
                Graph[i][j]['balance'] = capacity//2
                Graph[j][i]['balance'] = capacity - Graph[i][j]['balance']
            elif option == '99-1':
                coin = random.randint(0,1)
                if coin == 0:
                    Graph[i][j]['balance'] = round(0.99*capacity)
                    Graph[j][i]['balance'] = capacity - Graph[i][j]['balance']
                else:
                    Graph[j][i]['balance'] = round(0.99*capacity)
                    Graph[i][j]['balance'] = capacity - Graph[j][i]['balance']
            else:
                raise Exception ('No valid option to set balance selected')
    return Graph

def set_balance_ln (Graph, alpha: float = 0.01):
    number_nodes = round(Graph.number_of_nodes()*alpha)
    k_central_nodes = get_k_most_centralized_nodes (Graph, number_nodes)
    k_central_nodes_dict = dict.fromkeys(k_central_nodes, "True")

    for node in k_central_nodes:
        for neighbor in Graph.neighbors(node):
            if neighbor in k_central_nodes_dict:
                Graph[node][neighbor]['balance'] = int(Graph[node][neighbor]['capacity'])//2
                Graph[neighbor][node]['balance'] = int(Graph[node][neighbor]['capacity']) - Graph[node][neighbor]['balance'] 
            else:
                coin = random.randint(0,1)
                if coin == 0:
                    Graph[node][neighbor]['balance'] = round(0.99*int(Graph[node][neighbor]['capacity']))
                    Graph[neighbor][node]['balance'] = int(Graph[node][neighbor]['capacity']) - Graph[node][neighbor]['balance']
                else:
                    Graph[neighbor][node]['balance'] = round(0.99*int(Graph[node][neighbor]['capacity']))
                    Graph[node][neighbor]['balance'] = int(Graph[node][neighbor]['capacity']) - Graph[neighbor][node]['balance']
    
    for (i,j) in Graph.edges():
        if 'balance' not in Graph[i][j]:
            capacity = int(Graph[i][j]['capacity'])
            coin = random.randint(0,1)
            if coin == 0:
                Graph[i][j]['balance'] = round(0.99*capacity)
                Graph[j][i]['balance'] = capacity - Graph[i][j]['balance']
            else:
                Graph[j][i]['balance'] = round(0.99*capacity)
                Graph[i][j]['balance'] = capacity - Graph[j][i]['balance']
    return Graph

def find_shortest_path (Graph, s, t, value):
    return nx.shortest_path(Graph, source = s, target = t, weight = 'fee_base_msat' + (value*'fee_proportional_millionths'/1000000))

def make_payment (Graph, s, t, value, path = None, debug = False):
    if path == None:
        hops = nx.shortest_path(Graph, s, t)
    else:
        hops = path
    
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
                if debug == True:
                    print("\nBalance before channel" + str((hops[index-1],hops[index])) + ":" + str(Graph[hops[index-1]][hops[index]]['balance']))
                    print("Balance before channel" + str((hops[index],hops[index-1])) + ":" + str(Graph[hops[index]][hops[index-1]]['balance']))
                Graph[hops[index-1]][hops[index]]['balance'] = Graph[hops[index-1]][hops[index]]['balance'] - value
                Graph[hops[index]][hops[index-1]]['balance'] = Graph[hops[index]][hops[index-1]]['balance'] + value
                if debug == True:
                    print("Balance after channel" + str((hops[index-1],hops[index])) + ":" + str(Graph[hops[index-1]][hops[index]]['balance']))    
                    print("Balance after channel" + str((hops[index],hops[index-1])) + ":" + str(Graph[hops[index]][hops[index-1]]['balance']))
        except:
            raise Exception ('Could not issue payment')

def get_node_balance (Graph, node, debug = False):
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

       