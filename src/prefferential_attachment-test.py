from topology import *
from prefferential_attachment import *
from random import randint
import pickle

def greedy_algorithm_test (Graph, node_diff, cycle, k):

    nodes_file = open("../results/selected_nodes_greedy_" + str(cycle),"wb")
    reward_file = open("../results/reward_greedy_" + str(cycle),"wb")

    for node in node_diff:
        selected_node, reward = greedy_algorithm (Graph, node, k, cycle = cycle)
        pickle.dump(selected_node, nodes_file)
        pickle.dump(reward, reward_file)

def random_test(Graph, node_diff, cycle, k):
    nodes_file = open("../results/selected_nodes_random_" + str(cycle),"wb")
    reward_file = open("../results/reward_random_" + str(cycle),"wb")

    for node in node_diff:
        pdf = get_uniform_distribution_pdf(Graph)
        selected_node = sample_pdf(pdf, k)
        for node in selected_node:
            print('Node: ' + str(node))


def bc_test(Graph, node_diff, cycle, k):
    nodes_file = open("../results/selected_nodes_bc_" + str(cycle),"wb")
    reward_file = open("../results/reward_bc_" + str(cycle),"wb")

    for node in node_diff:
        pdf = get_centrality_distribution_pdf(Graph)
        selected_node = sample_pdf(pdf, k)
        payment_graph = make_graph_payment(Graph, 4104693)
        for node in selected_node:
            print('Node: ' + str(node))
            print('Node BC: ' + str(nx.betweenness_centrality(payment_graph, weight = 'fee')[node]))

def degree_test(Graph, node_diff, cycle, k):
    nodes_file = open("../results/selected_nodes_bc_" + str(cycle),"wb")
    reward_file = open("../results/reward_bc_" + str(cycle),"wb")

    for node in node_diff:
        pdf = get_degree_distribution_pdf(Graph)
        selected_node = sample_pdf(pdf, k)
        payment_graph = make_graph_payment(Graph, 4104693)
    for selected in selected_node:
        print('Node: ' + str(selected))
        print('Node degree: ' + str(Graph.degree()[selected]))
    reward, _ = add_selected_edges(Graph, selected_node, node, 0.5)
    print(reward)



#Graph = graph_names('jul 2022')
Graph = generate_graph(n=512, m= 3, option='barabasi-albert')
Graph = set_attributes(Graph, 'lightning')
Graph = validate_graph(Graph)
#Graph = snowball_sample(Graph, size=512)
node_diff = ['new_node']

#greedy_algorithm_test(Graph, node_diff, False, 4)
degree_test(Graph, node_diff, False, 4)
