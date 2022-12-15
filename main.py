from topology import *
from prefferential_attachment import *
from random import randint
import pickle

def main ():
    filenames, dates = generate_timestamps()
    Graph = create_graph(filenames[0])
    node_diff = list(get_node_diff(create_graph(filenames[1]), create_graph(filenames[0])))[-1:]

    transitivity_file = open("transitivity", "w")
    nodes_file = open("selected_nodes","wb")
    reward_file = open("reward","wb")

    for node in node_diff:
        transitivity = (nx.transitivity(Graph))
        transitivity_file.write(str(transitivity))
        transitivity_file.flush()
        selected_node, reward = incremental_closeness (Graph, node, 2)
        pickle.dump(selected_node, nodes_file)
        pickle.dump(reward, reward_file)

        transitivity = (nx.transitivity(Graph))
        transitivity_file.write(str(transitivity))
        transitivity_file.flush()
        selected_node, reward = incremental_closeness (Graph, node, 1)
        pickle.dump(selected_node, nodes_file)
        pickle.dump(reward, reward_file)
        
        transitivity = (nx.transitivity(Graph))
        transitivity_file.write(str(transitivity))
        transitivity_file.flush()
        selected_node, reward = incremental_closeness (Graph, node, 1)
        nodes_file.write(selected_node)
        reward_file.write(reward)
        
        transitivity = (nx.transitivity(Graph))
        transitivity_file.write(str(transitivity))
        transitivity_file.flush()
        selected_node, reward = incremental_closeness (Graph, node, 1)
        pickle.dump(selected_node, nodes_file)
        pickle.dump(reward, reward_file)

        transitivity = (nx.transitivity(Graph))
        transitivity_file.write(str(transitivity))
        transitivity_file.flush()
    
    nx.write_graphml(Graph, 'graph_after')
    transitivity_file.close()
    nodes_file.close()
    reward_file.close()

main()
