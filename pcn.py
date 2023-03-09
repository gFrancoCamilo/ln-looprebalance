from topology import *
import networkx as nx

def make_graph_directed (Graph):
    return Graph.to_directed()

def set_balance (Graph):
    edges_to_remove = []
    for (i,j) in Graph.edges:
        if 'htlc_maximum_msat' in Graph[i][j]:
            capacity = Graph[i][j]['htlc_maximum_msat']
            Graph[i][j]['balance'] = capacity/2
        else:
            edges_to_remove.append((i,j))
    
    remove_edges_without_data(edges_to_remove, Graph)
    return Graph

def remove_edges_without_data(edges_to_remove, Graph):
    try:
        for (i,j) in edges_to_remove:
            Graph.remove_edge(i,j)
    except:
        raise Exception ('Could not remove edge')

def find_shortest_path (Graph, s, t, value):
    return nx.shortest_path(Graph, source =s, target = t, weight = 'fee_base_msat' + (value*'fee_proportional_millionths'/1000000))

def make_payment (Graph, s, t, value, path = None, debug = False):
    if path == None:
        hops = nx.shortest_path(Graph, s, t)
    else:
        hops = path
    
    if len(path) == 0:
        raise Exception ('Path is empty')

    index = 0
    if debug == True:
        print("Hops: " + str(hops))
    
    while index < (len(hops) - 1):
        index += 1
        if value > Graph[hops[index-1]][hops[index]]['balance']:
            raise Exception ('Not enough balance on path')

    index = 0
    while index < (len(hops) - 1):
        index += 1
        try:
            if Graph[hops[index-1]][hops[index]]['balance'] - value < 0:
                raise Exception ('Out of funds')
            else:
                if debug == True:
                    print("\nBalanço antes canal" + str((hops[index-1],hops[index])) + ":" + str(Graph[hops[index-1]][hops[index]]['balance']))
                    print("Balanço antes canal" + str((hops[index],hops[index-1])) + ":" + str(Graph[hops[index]][hops[index-1]]['balance']))
                Graph[hops[index-1]][hops[index]]['balance'] = Graph[hops[index-1]][hops[index]]['balance'] - value
                Graph[hops[index]][hops[index-1]]['balance'] = Graph[hops[index]][hops[index-1]]['balance'] + value
                if debug == True:
                    print("Balanço depois canal" + str((hops[index-1],hops[index])) + ":" + str(Graph[hops[index-1]][hops[index]]['balance']))    
                    print("Balanço depois canal" + str((hops[index],hops[index-1])) + ":" + str(Graph[hops[index]][hops[index-1]]['balance']))
        except:
            #undo_payment (Graph, s, t, value, hops, index)
            raise Exception ('Could not issue payment')

def get_node_balance (Graph, node, debug = False):
    try:
        neighbors = [n for n in Graph.neighbors(node)]
        node_balance = 0
        for neighbor in neighbors:
            node_balance += Graph[node][neighbor]['balance']
        if debug == True:
            print("\nBalanço do nó " + str(node) + " : " + str(node_balance))
        return node_balance
    except:
        raise Exception("Could not compute node balance")


#def undo_payment (Graph, s, t, value, hops, index):
#    while index > 0:
       