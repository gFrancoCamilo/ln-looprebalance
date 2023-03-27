from pcn import *
from topology import *
import numpy as np

def set_balance_test (Graph, option: str = '99-1'):
    """
    set_balance_test tests the function set_balance in pcn.py.
    Given the chosen option, it verifies if the returned Graph
    initialized channel balances correctly.
    """
    Graph = set_balance(Graph)
    for (i,j) in Graph.edges:
        try:
            if 'balance' in Graph[i][j] == True: 
                assert Graph[i][j]['balance'] == Graph[j][i]['balance']
                if option == 'half':
                    assert Graph[i][j]['balance'] == int(Graph[i][j]['capacity'])//2
                if option == "99-1":
                    assert Graph[i][j]['balance'] == int(Graph[i][j]['capacity'])*0.99 or Graph[i][j]['balance'] == int(Graph[i][j]['capacity'])*0.01
        except:
            raise Exception ("Failed to set balance")

def make_payment_test (Graph, s,t, value):
    """
    make_payment_test tests the function make_payment from pcn.py.
    The function tries to issue a payment of a given value from a
    given source-destination pair.
    """
    Graph = set_balance(Graph)
    try:
        make_payment(Graph, s, t, value, debug = True)
    except:
        raise Exception ('Failed to test payment')

def set_balance_ln_test (Graph, alpha: float = 0.01, debug: bool = False):
    """
    set_balance_ln_test tests the function set_balance_ln in pcn.py.
    """
    try:
        Graph = set_balance_ln(Graph, alpha)
        if debug == True:
            for (i,j) in Graph.edges():
                print("Node i balance: " + str(Graph[i][j]['balance']))
                print("Node j balance: " + str(Graph[j][i]['balance']))
    except:
        raise Exception ("Failed to set ln balance")

Graph = graph_names ('jul 2022')
Graph = validate_graph(Graph)

v1 = np.random.choice(Graph.nodes())
v2 = np.random.choice(Graph.nodes())
print("Chosen source: " + str(v1))
print("Chosen destination: " + str(v2))

#set_balance_test(Graph)
set_balance_ln_test(Graph, 0.01)
make_payment_test(Graph, v1, v2, 100)
