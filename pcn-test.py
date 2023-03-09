from pcn import *
from topology import *
import numpy as np

def make_graph_directed_test (Graph):
    Graph = make_graph_directed (Graph)
    for (i,j) in Graph.edges:
        try:
            if 'htlc_maximum_msat' in Graph[i][j] == True:
                assert Graph[i][j]['htlc_maximum_msat'] == Graph[j][i]['htlc_maximum_msat']
        except:
            raise Exception ("Failed to make Graph directed")

def set_balance_test (Graph):
    Graph = make_graph_directed (Graph)
    Graph = set_balance(Graph)
    for (i,j) in Graph.edges:
        try:
            if 'balance' in Graph[i][j] == True: 
                assert Graph[i][j]['balance'] == Graph[j][i]['balance']
        except:
            raise Exception ("Failed to set balance")

def make_payment_test (Graph, s,t, value):
    Graph = make_graph_directed (Graph)
    Graph = set_balance(Graph)
    try:
        make_payment(Graph, s, t, value, debug = True)
    except:
        raise Exception ('Failed to test payment')

filenames, _ = generate_timestamps()
print(filenames[0])
Graph = create_graph (filenames[0])

v1 = np.random.choice(Graph.nodes())
v2 = np.random.choice(Graph.nodes())
print("Chosen source: " + str(v1))
print("Chosen destination: " + str(v2))

make_graph_directed_test(Graph)
set_balance_test(Graph)
make_payment_test(Graph, '029141fa00c43a0d3be9ab25a7fd44faa5d36272f8a8b9c1fe43bac1ae6e96128f', '0348449c971a98991295e568eaf7ffd9416e7f6d1708b9dd32c76e0e180e29d126', 100)
make_payment_test(Graph, v1, v2, 100)
