from topology import *
from results import *
from pcn import *
from prefferential_attachment import *
from tqdm import tqdm
from throughput import *
import click
import threading
import pickle
import uuid

@click.group(chain=True, invoke_without_command=True)
def cli():
    pass

@cli.command(name='success-ratio', help='Gets payment success ratio')
@click.option('-b','--balance', default='ln', type=click.Choice (['ln','99-1','half'],
             case_sensitive=False), help = 'Define initial balance used in the simulation')
@click.option('-b_parameter','--balance-parameter', default=0.1, type=float,
            help='Defines the percentage of highest degree node that will be balanced 50-50')
@click.option('-n_payments','--number_payments', type=int, default=100,
             help='Number of payments in the simualtion')
@click.option('-t','--topology', default='lightning',
             type=click.Choice (['scale-free','watts-strogatz','barabasi-albert', 'lightning'], case_sensitive=False),
             help = 'Graph topology used in the simulation')
@click.option('-n','--nodes', type=int, default=10, help='Number of nodes in the topology. Only used if topology is not lightning.')
@click.option('--alpha', default=0.5, help='Alpha parameter for scale-free topology. Only used with scale-free topology.')
@click.option('--beta', default=0.00001, help='Beta parameter for scale-free topology. Only used with scale-free topology.')
@click.option('--gamma', default=0.49999, help='Gamma parameter for scale-free topology. Only used with scale-free topology.')
@click.option('-k', default=2, help='K parameter for Watts-Strogatz graph. Only used with Watts-Strogatz topology.')
@click.option('-p', default=0.1, help='P parameter for Watts-Strogatz graph. Only used with Watts-Strogatz topology.')
@click.option('-m', default=2, help='M parameter for Barabasi-Albert graph. Only used with Watts-Strogatz topology.')
@click.option('-d', '--date', default='jul 2022', type=click.Choice(['jul 2021', 'jan 2022', 'jul 2022'], case_sensitive=False),
            help='Date of lighting snapshot to be used in the simulation. Only used with lightning topology.')
@click.option('-pay','--payment_method', default='ripple', type=click.Choice(['ripple','credit-card'], case_sensitive=False),
            help='Dataset used to simulate payment in the network.')
@click.option('--lnd', is_flag=True, default=True, help='Payment routing follows a trial-and-error model')
def simulate_success_ratio (balance, balance_parameter, number_payments, topology, nodes, alpha, beta, gamma, k, p, m, date, payment_method, lnd):
    results = []
    if topology == 'lightning':
        Graph = graph_names(date)
        if balance != 'ln':
            Graph = set_balance(Graph, balance)
        else:
            Graph = set_balance_ln(Graph, balance_parameter)
        payment_dataset = get_payment_dataset(payment_method)
        payment_list = choose_payments(payment_dataset, number_payments)
        payment_dict = get_payments_ln(Graph, payment_list)
        event = threading.Event()
        return get_success_ratio(Graph, payment_dict, lnd=lnd)

@cli.command(name='channel-throughput', help='Gets payment success ratio')
@click.option('-b','--balance', default='ln', type=click.Choice (['ln','99-1','half'],
             case_sensitive=False), help = 'Define initial balance used in the simulation')
@click.option('-b_parameter','--balance-parameter', default=0.1, type=float,
            help='Defines the percentage of highest degree node that will be balanced 50-50')
@click.option('-n_payments','--number_payments', type=int, default=100,
             help='Number of payments in the simualtion')
@click.option('-t','--topology', default='lightning',
             type=click.Choice (['scale-free','watts-strogatz','barabasi-albert', 'lightning'], case_sensitive=False),
             help = 'Graph topology used in the simulation')
@click.option('-n','--nodes', type=int, default=10, help='Number of nodes in the topology. Only used if topology is not lightning.')
@click.option('--alpha', default=0.5, help='Alpha parameter for scale-free topology. Only used with scale-free topology.')
@click.option('--beta', default=0.00001, help='Beta parameter for scale-free topology. Only used with scale-free topology.')
@click.option('--gamma', default=0.49999, help='Gamma parameter for scale-free topology. Only used with scale-free topology.')
@click.option('-k', default=2, help='K parameter for Watts-Strogatz graph. Only used with Watts-Strogatz topology.')
@click.option('-p', default=0.1, help='P parameter for Watts-Strogatz graph. Only used with Watts-Strogatz topology.')
@click.option('-m', default=2, help='M parameter for Barabasi-Albert graph. Only used with Watts-Strogatz topology.')
@click.option('-d', '--date', default='jul 2022', type=click.Choice(['jul 2021', 'jan 2022', 'jul 2022'], case_sensitive=False),
            help='Date of lighting snapshot to be used in the simulation. Only used with lightning topology.')
@click.option('-pay','--payment_method', default='ripple', type=click.Choice(['ripple','credit-card'], case_sensitive=False),
            help='Dataset used to simulate payment in the network.')
@click.option('--lnd', is_flag=True, default=True, help='Payment routing follows a trial-and-error model')
def get_channel_troughput (balance, balance_parameter, number_payments, topology, nodes, alpha, beta, gamma, k, p, m, date, payment_method, lnd):
    results = []
    if topology == 'lightning':
        Graph = graph_names(date)
        if balance != 'ln':
            Graph = set_balance(Graph, balance)
        else:
            Graph = set_balance_ln(Graph, balance_parameter)
        payment_dataset = get_payment_dataset(payment_method)
        payment_list = choose_payments(payment_dataset, number_payments)
        payment_dict = get_payments_ln(Graph, payment_list)
        event = threading.Event()
        edges = [('038fe1bd966b5cb0545963490c631eaa1924e2c4c0ea4e7dcb5d4582a1e7f2f1a5', '02f3069a342ae2883a6f29e275f06f28a56a6ea2e2d96f5888a3266444dcf542b6'),
                ('02ec921faa245b5813385e04a51a6f0e2d99a570ff820bc42b6d1048c278f21216', '035e4ff418fc8b5554c5d9eea66396c227bd429a3251c8cbc711002ba215bfc226'),
                ('03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f', '03dc686001f9b1ff700dfb8917df70268e1919433a535e1fb0767c19223509ab57'),
                ('03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f', '02634002b0b3469695a3cd5a29e363071a09374c7753b3aea34d1fc297226b4dba'),
                ('03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f', '03683e95c542d5aae980b2cb346b6e5ff2416222b6962cf3f23dfbf4f93b409643'),
                ('03c8dfbf829eaeb0b6dab099d87fdf7f8faceb0c1b935cd243e8c1fb5af71361cf', '03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f'),
                ('026165850492521f4ac8abd9bd8088123446d126f648ca35e60f88177dc149ceb2', '03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f'),
                ('03c8dfbf829eaeb0b6dab099d87fdf7f8faceb0c1b935cd243e8c1fb5af71361cf', '022755c3ff4e5a1d71f573cda4b315887fc00a9e5c9ea9a847d939f3e517e69a70')]
        
        for edge in edges:
            init_window(Graph, edge, event, 10)
        get_success_ratio(Graph, payment_dict, lnd=lnd)
        event.set()

        for (i,j) in Graph.edges():
            if len(Graph[i][j]['payments']) > 2:
                print(str((i,j)) + ': ' + str(Graph[i][j]['payments']))

@cli.command(name='node-attachment', help='Gets node attachment results')
@click.option('-t','--topology', default='lightning',
             type=click.Choice (['watts-strogatz','barabasi-albert', 'lightning'], case_sensitive=False),
             help = 'Graph topology used in the simulation')
@click.option('-n','--nodes', type=int, default=512, help='Number of nodes in the topology.')
@click.option('-k', default=2, help='K parameter for Watts-Strogatz graph. Only used with Watts-Strogatz topology.')
@click.option('-p', default=0.1, help='P parameter for Watts-Strogatz graph. Only used with Watts-Strogatz topology.')
@click.option('-m', type=int,default=2, help='M parameter for Barabasi-Albert graph. Only used with Watts-Strogatz topology.')
@click.option('-d', '--date', default='jul 2022', type=click.Choice(['jul 2021', 'jan 2022', 'jul 2022'], case_sensitive=False),
            help='Date of lighting snapshot to be used in the simulation. Only used with lightning topology.')
@click.option('-c','--channels', type=int, default = 5, help='Number of channels to create.')
@click.option('-a', '--alpha', type=float, default = 0.5, help='Alpha parameter for reward computation.')
@click.option('--cycle', is_flag=True, default=False, help='Greedy attachment strategy focuses on cycle creation.')
def node_attachment (topology, nodes, k, p, m, date, channels, alpha, cycle):
    if topology == 'lightning':
        print('Generating Lightning Graph...')
        Graph = graph_names(date)
        print('Snowball sampling Lightning Graph...')
        Graph = snowball_sample(Graph, size = nodes)
    elif topology == 'watts-strogatz':
        print('Generating Watts-Strogatz Graph...')
        Graph = generate_graph(nodes, k=k, p=p, option='watts-strogatz') 
        print('Setting attributes...')   
        Graph = set_attributes(Graph, 'lightning')
    elif topology == 'barabasi-albert':
        print('Generating Barabasi-Albert Graph...')
        Graph = generate_graph(nodes, m=m, option='barabasi-albert') 
        print('Setting attributes...')
        Graph = set_attributes(Graph, 'lightning')
    
    Graph = validate_graph(Graph)
    node = 'new_node'
    nx.write_gml(Graph, '../results/node_attachment_results/graphs/'+topology+str(uuid.uuid4())+'_'+str(cycle)+'_alpha_'+str(alpha)+'.gml')

    print('Adding new node to topology using the greedy algorithm')
    selected_node_greedy, greedy_reward = greedy_algorithm(Graph.copy(), node, channels, alpha, cycle)
    greedy_fp = open('../results/node_attachment_results/greedy_' + str(cycle) + '_' + topology + '_alpha_'+str(alpha)+'.dat', 'ab+')
    
    print('Adding new nodes to topology using uniform distribution...')
    pdf = get_uniform_distribution_pdf(Graph)
    selected_node_random = sample_pdf(pdf, channels)
    random_reward, _ = add_selected_edges(Graph.copy(), selected_node_random, node, alpha)

    print('Adding new nodes to topology using centrality distribution...')
    pdf = get_centrality_distribution_pdf(Graph)
    selected_node_centrality = sample_pdf(pdf, channels)
    centrality_reward, _ = add_selected_edges(Graph.copy(), selected_node_centrality, node, alpha)

    print('Adding new nodes to topology using degree distribution...')
    pdf = get_degree_distribution_pdf(Graph)
    selected_node_degree = sample_pdf(pdf, channels)
    degree_reward, _ = add_selected_edges(Graph.copy(), selected_node_degree, node, alpha)

    print('Adding new nodes to topology using richest nodes distribution...')
    pdf = get_rich_nodes_pdf(Graph)
    selected_node_rich = sample_pdf(pdf, channels)
    rich_reward, _ = add_selected_edges(Graph.copy(), selected_node_rich, node, alpha)

    random_fp = open('../results/node_attachment_results/random_' + str(cycle) + '_' + topology + '_alpha_'+str(alpha)+'.dat', 'ab+')
    centrality_fp = open('../results/node_attachment_results/centrality_' + str(cycle) + '_' + topology + '_alpha_'+str(alpha)+'.dat', 'ab+')
    degree_fp = open('../results/node_attachment_results/degree_' + str(cycle) + '_' + topology + '_alpha_'+str(alpha)+'.dat', 'ab+')
    rich_fp = open('../results/node_attachment_results/rich_' + str(cycle) + '_' + topology + '_alpha_'+str(alpha)+'.dat', 'ab+')

    pickle.dump((greedy_reward, selected_node_greedy), greedy_fp)
    pickle.dump((random_reward, list(selected_node_random)), random_fp)
    pickle.dump((centrality_reward, list(selected_node_centrality)), centrality_fp)
    pickle.dump((degree_reward, list(selected_node_degree)), degree_fp)
    pickle.dump((rich_reward, list(selected_node_rich)), rich_fp)

    greedy_fp.close()
    random_fp.close()
    centrality_fp.close()
    degree_fp.close()
    rich_fp.close()

@cli.command(name='attachment-multiple-nodes', help='Gets node attachment results')
@click.option('-t','--topology', default='lightning',
             type=click.Choice (['watts-strogatz','barabasi-albert', 'lightning'], case_sensitive=False),
             help = 'Graph topology used in the simulation')
@click.option('-s','--snowball_size', type=int, default=512, help='Number of nodes in the topology be it snowball sampled or synthetic topology created.')
@click.option('-k', default=2, help='K parameter for Watts-Strogatz graph. Only used with Watts-Strogatz topology.')
@click.option('-p', default=0.1, help='P parameter for Watts-Strogatz graph. Only used with Watts-Strogatz topology.')
@click.option('-m', type=int,default=2, help='M parameter for Barabasi-Albert graph. Only used with Watts-Strogatz topology.')
@click.option('-d', '--date', default='jul 2022', type=click.Choice(['jul 2021', 'jan 2022', 'jul 2022'], case_sensitive=False),
            help='Date of lighting snapshot to be used in the simulation. Only used with lightning topology.')
@click.option('-c','--channels', type=int, default = 3, help='Number of channels to create.')
@click.option('-a', '--alpha', type=float, default = 0.5, help='Alpha parameter for reward computation.')
@click.option('--cycle', is_flag=True, default=True, help='Greedy attachment strategy focuses on cycle creation.')
@click.option('-n', '--nodes', default = 5, type = int, help='Number of nodes to add to the topology')
def attachment_multiple_nodes (topology, snowball_sizes, k, p, m, date, channels, alpha, cycle, nodes):
    if topology == 'lightning':
        print('Generating Lightning Graph...')
        Graph = graph_names(date)
        print('Snowball sampling Lightning Graph...')
        Graph = snowball_sample(Graph, size = snowball_sizes)
    elif topology == 'watts-strogatz':
        print('Generating Watts-Strogatz Graph...')
        Graph = generate_graph(snowball_sizes, k=k, p=p, option='watts-strogatz') 
        print('Setting attributes...')   
        Graph = set_attributes(Graph, 'lightning')
    elif topology == 'barabasi-albert':
        print('Generating Barabasi-Albert Graph...')
        Graph = generate_graph(snowball_sizes, m=m, option='barabasi-albert') 
        print('Setting attributes...')
        Graph = set_attributes(Graph, 'lightning')
    
    Graph = validate_graph(Graph)
    if topology != 'lightning':
        nx.write_gml(Graph, '../results/node_attachment_results/multiple_nodes/graphs/'+topology+str(uuid.uuid4())+'_'+str(cycle)+'_alpha_'+str(alpha)+'.gml')

    print('Adding new node to topology using the greedy algorithm')

    greedy_fp = open('../results/node_attachment_results/multiple_nodes/greedy_' + str(cycle) + '_' + topology + '_alpha_'+str(alpha)+'.dat', 'ab+')
    random_fp = open('../results/node_attachment_results/multiple_nodes/random_' + str(cycle) + '_' + topology + '_alpha_'+str(alpha)+'.dat', 'ab+')
    centrality_fp = open('../results/node_attachment_results/multiple_nodes/centrality_' + str(cycle) + '_' + topology + '_alpha_'+str(alpha)+'.dat', 'ab+')
    degree_fp = open('../results/node_attachment_results/multiple_nodes/degree_' + str(cycle) + '_' + topology + '_alpha_'+str(alpha)+'.dat', 'ab+')
    rich_fp = open('../results/node_attachment_results/multiple_nodes/rich_' + str(cycle) + '_' + topology + '_alpha_'+str(alpha)+'.dat', 'ab+')

    greedy_copy = Graph.copy()
    random_copy = Graph.copy()
    centrality_copy = Graph.copy()
    degree_copy = Graph.copy()
    rich_copy = Graph.copy()

    for node in range(nodes):
        selected_node_greedy, greedy_reward = greedy_algorithm(greedy_copy, node, channels, alpha, cycle)
        pickle.dump((greedy_reward,list(selected_node_greedy)), greedy_fp)

        print('Adding new nodes to topology using uniform distribution...')
        pdf = get_uniform_distribution_pdf(Graph)
        selected_node_random = sample_pdf(pdf, channels)
        random_reward, _ = add_selected_edges(random_copy, selected_node_random, node, alpha)
        pickle.dump((random_reward,list(selected_node_random)), random_fp)

        print('Adding new nodes to topology using centrality distribution...')
        pdf = get_centrality_distribution_pdf(Graph)
        selected_node_centrality = sample_pdf(pdf, channels)
        centrality_reward, _ = add_selected_edges(centrality_copy, selected_node_centrality, node, alpha)
        pickle.dump((centrality_reward,list(selected_node_centrality)), centrality_fp)

        print('Adding new nodes to topology using degree distribution...')
        pdf = get_degree_distribution_pdf(Graph)
        selected_node_degree = sample_pdf(pdf, channels)
        degree_reward, _ = add_selected_edges(degree_copy, selected_node_degree, node, alpha)
        pickle.dump((degree_reward,list(selected_node_degree)), degree_fp)

        print('Adding new nodes to topology using richest nodes distribution...')
        pdf = get_rich_nodes_pdf(Graph)
        selected_node_rich = sample_pdf(pdf, channels)
        rich_reward, _ = add_selected_edges(rich_copy, selected_node_rich, node, alpha)
        pickle.dump((rich_reward,list(selected_node_rich)), rich_fp)

            
    greedy_fp.close()
    random_fp.close()
    centrality_fp.close()
    degree_fp.close()
    rich_fp.close()
    
    print('Number of node in greedy graph: ' + str(greedy_copy.number_of_nodes()))
    print('Number of node in random graph: ' + str(random_copy.number_of_nodes()))
    print('Number of node in centrality graph: ' + str(centrality_copy.number_of_nodes()))
    print('Number of node in degree graph: ' + str(degree_copy.number_of_nodes()))
    print('Number of node in rich graph: ' + str(rich_copy.number_of_nodes()))

    
    


if __name__ == '__main__':
    cli()
