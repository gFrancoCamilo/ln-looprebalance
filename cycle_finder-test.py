from cycle_finder import *
from topology import *
from pcn import *
import networkx as nx
import numpy as np

def find_cycle_test (Graph, node = None, channel = None, debug = False):
    """
    find_cycle_test tests the function find_cycle in cycle_finder.py.
    If no node or channel is passed to test, the function randomly 
    choses a node from the graph and a channel from its neighbors 
    """
    try:
        print("--------------Results of cycle_to_path_test---------------\n")
        if node == None:
            node = np.random.choice(Graph.nodes())
        if channel == None:
            neighbors = [n for n in Graph[node]]
            if debug == True:
                print("Neighbors: " + str(neighbors))
            second = np.random.choice(neighbors)
            if debug == True:
                print ("Dest chosen: " + str(second))
            channel = (node, second)
        if debug == True:
            print ("Node: " + str(node))
            print ("Channel: " + str(channel))

        node_balance_before = get_node_balance(Graph, node)
        cycles = find_cycle (Graph, channel, node)
        node_balance_after = get_node_balance(Graph, node)

        assert node_balance_before == node_balance_after

        print ("Cycles: " + str(cycles))
        return node, cycles
    except:
        raise Exception ("Error finding cycles")

filenames, _ = generate_timestamps()
print(filenames[0])
Graph = create_graph (filenames[0])
Graph = make_graph_directed(Graph)
set_balance(Graph)
random_node, cycles = find_cycle_test(Graph, debug = True)
make_payment(Graph, random_node, random_node, 100, path = cycles[0], debug = True)


#for cycle in nx.simple_cycles(Graph):
#    if random_node in cycle:
#        print(cycle)
