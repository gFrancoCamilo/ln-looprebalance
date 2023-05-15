from topology import *
from prefferential_attachment import *
from random import randint
import pickle

def main (result, cycle):
    Graph = graph_names('jul 2022')
    Graph = validate_graph(Graph)
    Graph = snowball_sample(Graph, size=512)
    node_diff = ['new_node']

    #transitivity_file = open("transitivity_" + result + "_" + str(cycle), "w")
    nodes_file = open("../results/selected_nodes_" + result + "_" + str(cycle),"wb")
    reward_file = open("../results/reward_" + result + "_" + str(cycle),"wb")

    if result == 'greedy':
        for node in node_diff:
            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()
            selected_node, reward = incremental_closeness (Graph, node, 5, cycle = cycle)
            pickle.dump(selected_node, nodes_file)
            pickle.dump(reward, reward_file)

    if result == 'degree':
        for node in node_diff:
            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()
            selected_node, reward = degree_only (Graph, node, 2, 'degree', cycle = cycle)
            pickle.dump(selected_node, nodes_file)
            pickle.dump(reward, reward_file)

            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()
            selected_node, reward = degree_only (Graph, node, 1, 'degree', cycle = cycle)
            pickle.dump(selected_node, nodes_file)
            pickle.dump(reward, reward_file)

            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()
            selected_node, reward = degree_only (Graph, node, 1, 'degree', cycle = cycle)
            pickle.dump(selected_node, nodes_file)
            pickle.dump(reward, reward_file)

            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()
            selected_node, reward = degree_only (Graph, node, 1, 'degree', cycle = cycle)
            pickle.dump(selected_node, nodes_file)
            pickle.dump(reward, reward_file)

            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()

    if result == 'bc':
        for node in node_diff:
            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()
            selected_node, reward = degree_only (Graph, node, 2, 'bc', cycle = cycle)
            pickle.dump(selected_node, nodes_file)
            pickle.dump(reward, reward_file)

            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()
            selected_node, reward = degree_only (Graph, node, 1, 'bc', cycle = cycle)
            pickle.dump(selected_node, nodes_file)
            pickle.dump(reward, reward_file)

            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()
            selected_node, reward = degree_only (Graph, node, 1, 'bc', cycle = cycle)
            pickle.dump(selected_node, nodes_file)
            pickle.dump(reward, reward_file)

            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()
            selected_node, reward = degree_only (Graph, node, 1, 'bc', cycle = cycle)
            pickle.dump(selected_node, nodes_file)
            pickle.dump(reward, reward_file)

            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()

    if result == 'cc':
        for node in node_diff:
            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()
            selected_node, reward = degree_only (Graph, node, 2, 'cc', cycle = cycle)
            pickle.dump(selected_node, nodes_file)
            pickle.dump(reward, reward_file)

            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()
            selected_node, reward = degree_only (Graph, node, 1, 'cc', cycle = cycle)
            pickle.dump(selected_node, nodes_file)
            pickle.dump(reward, reward_file)

            #ransitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()
            selected_node, reward = degree_only (Graph, node, 1, 'cc', cycle = cycle)
            pickle.dump(selected_node, nodes_file)
            pickle.dump(reward, reward_file)

            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()
            selected_node, reward = degree_only (Graph, node, 1, 'cc', cycle = cycle)
            pickle.dump(selected_node, nodes_file)
            pickle.dump(reward, reward_file)

            #transitivity = (nx.transitivity(Graph))
            #transitivity_file.write(str(transitivity))
            #transitivity_file.flush()


    #nx.write_graphml(Graph, 'graph_after_'+result + "_" + str(cycle))
    #transitivity_file.close()
    nodes_file.close()
    reward_file.close()

main('greedy', False)
