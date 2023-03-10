from pcn import *
from topology import *
import numpy as np

def set_balance_test (Graph, option: str = '99-1'):
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
    Graph = set_balance(Graph)
    try:
        make_payment(Graph, s, t, value, debug = True)
    except:
        raise Exception ('Failed to test payment')

Graph = graph_names ('jul 2022')
Graph = validate_graph(Graph)

v1 = np.random.choice(Graph.nodes())
v2 = np.random.choice(Graph.nodes())
print("Chosen source: " + str(v1))
print("Chosen destination: " + str(v2))

set_balance_test(Graph)
make_payment_test(Graph, v1, v2, 100)
