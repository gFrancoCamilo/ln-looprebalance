from topology import *
from prefferential_attachment import *
from random import randint

def main ():
    filenames, dates = generate_timestamps()
    Graph = create_graph(filenames[0])
    node_diff = get_node_diff(create_graph(filenames[1]), Graph)

    transitivity = []

    for node in node_diff:
        transitivity.append(nx.transitivity(Graph))
        incremental_closeness (Graph, node, randint(2,6))
        transitivity.append(nx.transitivity(Graph))
    
    transitivity_file = open("transitivity", "w")
    transitivity_file.write(transitivity)
    transitivity_file.close()
