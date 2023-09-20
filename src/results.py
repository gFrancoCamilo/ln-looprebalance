from topology import *
from pcn import *
from payments import *
from cycle_finder import *

import os
import pickle
import random as rnd

import networkx as nx
import numpy as np
import scipy.stats as st
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-colorblind')

def get_success_ratio (Graph: nx.DiGraph, payment_dict: dict, lnd: bool = True, debug: bool = False):
    """
    get_success_ratio gets the payment success ratio of the network for a given scenario.
    The function receives the network directed-Graph with balances already established
    and a payment dictionary with (source, destination) as key and payment value as
    value. The function calculates how many payments were successful out of the total
    and returns the ratio.
    """
    total_payments = 0
    successful_payments = 0
    desc = 'Issuing payments'
    for (i,j) in tqdm(payment_dict, desc):
        try:
            if lnd == True:
                make_payment_lnd(Graph, i, j, payment_dict[i,j])
            else:
                make_payment(Graph, i, j, payment_dict[i,j])
            successful_payments += 1
            total_payments += 1
        except Exception as e:
            total_payments += 1
    return successful_payments/total_payments

def plot_rewards (alpha, cycle= False):
    topologies = ['lightning','barabasi-albert','watts-strogatz']
    algorithms = ['greedy', 'centrality', 'degree', 'rich', 'random']
    
    lightning = []
    ba = []
    ws = []

    for topology in topologies:
        for heuristic in algorithms:
            fp = open('../results/node_attachment_results/'+heuristic+'_'+str(cycle)+'_'+topology+'_alpha_'+str(alpha)+'.dat','rb')
            while True:
                try:
                    (aux, _) = pickle.load(fp)
                    if topology == 'lightning':
                        lightning.append(aux)
                    if topology == 'barabasi-albert':
                        ba.append(aux)
                    if topology == 'watts-strogatz':
                        ws.append(aux)
                except EOFError:
                    break

    greedy = []
    centrality = []
    degree = []
    rich = []
    random = []
    counter = 0
    for element in lightning:
        if len(element) < 10:
            for index in range(len(element), 10):
                element.append(element[-1])
        counter += 1
        if counter <= 10:
            greedy.append(element)
        elif counter <= 20:
            centrality.append(element)
        elif counter <= 30:
            degree.append(element)
        elif counter <= 40:
            rich.append(element)
        elif counter <= 50:
            random.append(element)
    
    x = [number for number in range(1,11)]
    
    greedy = [list(a) for a in (zip(*greedy))]
    centrality = [list(a) for a in (zip(*centrality))]
    degree = [list(a) for a in (zip(*degree))]
    rich = [list(a) for a in (zip(*rich))]
    random = [list(a) for a in (zip(*random))]

    greedy_max = []
    greedy_min = []
    greedy_mean = []
    for element in greedy:
        greedy_mean.append(np.mean(element))
        greedy_max.append(1.96*np.std(element)/np.sqrt(10))
        greedy_min.append(1.96*np.std(element)/np.sqrt(10))
    greedy_err = [greedy_min, greedy_max]
    plt.errorbar(x, greedy_mean, yerr=greedy_err, label='ProfitPilot', lw=2)

    centrality_max = []
    centrality_min = []
    centrality_mean = []
    for element in centrality:
        centrality_mean.append(np.mean(element))
        centrality_max.append(1.96*np.std(element)/np.sqrt(10))
        centrality_min.append(1.96*np.std(element)/np.sqrt(10))
    centrality_err = [centrality_min, centrality_max]
    plt.errorbar(x, centrality_mean, yerr=centrality_err, ls = 'dashed', label='Centrality', lw=2)

    degree_max = []
    degree_min = []
    degree_mean = []
    for element in degree:
        degree_mean.append(np.mean(element))
        degree_max.append(1.96*np.std(element)/np.sqrt(10))
        degree_min.append(1.96*np.std(element)/np.sqrt(10))
    degree_err = [degree_min, degree_max]
    plt.errorbar(x, degree_mean, yerr=degree_err, ls = 'dotted', label='Degree', lw=2)

    rich_max = []
    rich_min = []
    rich_mean = []
    for element in rich:
        rich_mean.append(np.mean(element))
        rich_max.append(1.96*np.std(element)/np.sqrt(10))
        rich_min.append(1.96*np.std(element)/np.sqrt(10))
    rich_err = [rich_min, rich_max]
    plt.errorbar(x, rich_mean, yerr=rich_err, ls='dashdot', label='Rich', lw=2)

    random_max = []
    random_min = []
    random_mean = []
    for element in random:
        random_mean.append(np.mean(element))
        random_max.append(1.96*np.std(element)/np.sqrt(10))
        random_min.append(1.96*np.std(element)/np.sqrt(10))
    random_err = [random_min, random_max]
    plt.errorbar(x, random_mean, yerr=random_err, ls=(0, (3, 1, 1, 1, 1, 1)), label='Random', lw=2)
    plt.legend(fontsize=16)

    plt.ylabel('Incentive', fontsize=24)
    plt.xlabel('\# of Neighbors', fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)

    plt.tight_layout()
    plt.grid()
    plt.savefig('../results/node_attachment_results/lightning_'+str(cycle)+''+str(alpha)+'.pdf', dpi=600)
    plt.clf()

    greedy = []
    centrality = []
    degree = []
    rich = []
    random = []
    counter = 0
    for element in ba:
        if len(element) < 10:
            for index in range(len(element), 10):
                element.append(element[-1])
        counter += 1
        if counter <= 10:
            greedy.append(element)
        elif counter <= 20:
            centrality.append(element)
        elif counter <= 30:
            degree.append(element)
        elif counter <= 40:
            rich.append(element)
        elif counter <= 50:
            random.append(element)
        
    x = [number for number in range(1,11)]
    
    greedy = [list(a) for a in (zip(*greedy))]
    centrality = [list(a) for a in (zip(*centrality))]
    degree = [list(a) for a in (zip(*degree))]
    rich = [list(a) for a in (zip(*rich))]
    random = [list(a) for a in (zip(*random))]

    greedy_max = []
    greedy_min = []
    greedy_mean = []
    for element in greedy:
        greedy_mean.append(np.mean(element))
        greedy_max.append(1.96*np.std(element)/np.sqrt(10))
        greedy_min.append(1.96*np.std(element)/np.sqrt(10))
    greedy_err = [greedy_min, greedy_max]
    plt.errorbar(x, greedy_mean, yerr=greedy_err, label='ProfitPilot', lw=2)

    centrality_max = []
    centrality_min = []
    centrality_mean = []
    for element in centrality:
        centrality_mean.append(np.mean(element))
        centrality_max.append(1.96*np.std(element)/np.sqrt(10))
        centrality_min.append(1.96*np.std(element)/np.sqrt(10))
    centrality_err = [centrality_min, centrality_max]
    plt.errorbar(x, centrality_mean, yerr=centrality_err, ls = 'dashed', label='Centrality', lw=2)

    degree_max = []
    degree_min = []
    degree_mean = []
    for element in degree:
        degree_mean.append(np.mean(element))
        degree_max.append(1.96*np.std(element)/np.sqrt(10))
        degree_min.append(1.96*np.std(element)/np.sqrt(10))
    degree_err = [degree_min, degree_max]
    plt.errorbar(x, degree_mean, yerr=degree_err, ls = 'dotted', label='Degree', lw=2)

    rich_max = []
    rich_min = []
    rich_mean = []
    for element in rich:
        rich_mean.append(np.mean(element))
        rich_max.append(1.96*np.std(element)/np.sqrt(10))
        rich_min.append(1.96*np.std(element)/np.sqrt(10))
    rich_err = [rich_min, rich_max]
    plt.errorbar(x, rich_mean, yerr=rich_err, ls='dashdot', label='Rich', lw=2)

    random_max = []
    random_min = []
    random_mean = []
    for element in random:
        random_mean.append(np.mean(element))
        random_max.append(1.96*np.std(element)/np.sqrt(10))
        random_min.append(1.96*np.std(element)/np.sqrt(10))
    random_err = [random_min, random_max]
    plt.errorbar(x, random_mean, yerr=random_err, ls=(0, (3, 1, 1, 1, 1, 1)), label='Random', lw=2)
    plt.legend(fontsize=16)

    plt.ylabel('Incentive', fontsize=24)
    plt.xlabel('\# of Neighbors', fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)

    plt.tight_layout()
    plt.grid()
    plt.savefig('../results/node_attachment_results/ba_'+str(cycle)+''+str(alpha)+'.pdf', dpi=600)
    plt.clf()

    greedy = []
    centrality = []
    degree = []
    rich = []
    random = []
    counter = 0
    for element in ws:
        if len(element) < 10:
            for index in range(len(element), 10):
                element.append(element[-1])
        counter += 1
        if counter <= 10:
            greedy.append(element)
        elif counter <= 20:
            centrality.append(element)
        elif counter <= 30:
            degree.append(element)
        elif counter <= 40:
            rich.append(element)
        elif counter <= 50:
            random.append(element)
        
    x = [number for number in range(1,11)]
    
    greedy = [list(a) for a in (zip(*greedy))]
    centrality = [list(a) for a in (zip(*centrality))]
    degree = [list(a) for a in (zip(*degree))]
    rich = [list(a) for a in (zip(*rich))]
    random = [list(a) for a in (zip(*random))]

    greedy_max = []
    greedy_min = []
    greedy_mean = []
    for element in greedy:
        greedy_mean.append(np.mean(element))
        greedy_max.append(1.96*np.std(element)/np.sqrt(10))
        greedy_min.append(1.96*np.std(element)/np.sqrt(10))
    greedy_err = [greedy_min, greedy_max]
    plt.errorbar(x, greedy_mean, yerr=greedy_err, label='ProfitPilot', lw=2)

    centrality_max = []
    centrality_min = []
    centrality_mean = []
    for element in centrality:
        centrality_mean.append(np.mean(element))
        centrality_max.append(1.96*np.std(element)/np.sqrt(10))
        centrality_min.append(1.96*np.std(element)/np.sqrt(10))
    centrality_err = [centrality_min, centrality_max]
    plt.errorbar(x, centrality_mean, yerr=centrality_err, ls = 'dashed', label='Centrality', lw=2)

    degree_max = []
    degree_min = []
    degree_mean = []
    for element in degree:
        degree_mean.append(np.mean(element))
        degree_max.append(1.96*np.std(element)/np.sqrt(10))
        degree_min.append(1.96*np.std(element)/np.sqrt(10))
    degree_err = [degree_min, degree_max]
    plt.errorbar(x, degree_mean, yerr=degree_err, ls = 'dotted', label='Degree', lw=2)

    rich_max = []
    rich_min = []
    rich_mean = []
    for element in rich:
        rich_mean.append(np.mean(element))
        rich_max.append(1.96*np.std(element)/np.sqrt(10))
        rich_min.append(1.96*np.std(element)/np.sqrt(10))
    rich_err = [rich_min, rich_max]
    plt.errorbar(x, rich_mean, yerr=rich_err, ls='dashdot', label='Rich', lw=2)

    random_max = []
    random_min = []
    random_mean = []
    for element in random:
        random_mean.append(np.mean(element))
        random_max.append(1.96*np.std(element)/np.sqrt(10))
        random_min.append(1.96*np.std(element)/np.sqrt(10))
    random_err = [random_min, random_max]
    plt.errorbar(x, random_mean, yerr=random_err, ls=(0, (3, 1, 1, 1, 1, 1)), label='Random', lw=2)
    plt.legend(fontsize=16)

    plt.ylabel('Incentive', fontsize=24)
    plt.xlabel('\# of Neighbors', fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)

    plt.tight_layout()
    plt.grid()
    plt.savefig('../results/node_attachment_results/ws_'+str(cycle)+''+str(alpha)+'.pdf', dpi=600)

def compare_rewards_cycle (alpha=0.5):
    topologies = ['lightning','barabasi-albert','watts-strogatz']
    
    lightning = []
    ba = []
    ws = []
    lightning_True = []
    ba_True = []
    ws_True = []

    for topology in topologies:
        fp = open('../results/node_attachment_results/greedy_False_'+topology+'_alpha_'+str(alpha)+'.dat','rb')
        while True:
            try:
                (aux, _) = pickle.load(fp)
                if topology == 'lightning':
                    lightning.append(aux)
                if topology == 'barabasi-albert':
                    ba.append(aux)
                if topology == 'watts-strogatz':
                    ws.append(aux)
            except EOFError:
                break
        fp = open('../results/node_attachment_results/greedy_True_'+topology+'_alpha_'+str(alpha)+'.dat','rb')
        while True:
            try:
                (aux, _) = pickle.load(fp)
                if topology == 'lightning':
                    lightning_True.append(aux)
                if topology == 'barabasi-albert':
                    ba_True.append(aux)
                if topology == 'watts-strogatz':
                    ws_True.append(aux)
            except EOFError:
                break


    for topology in topologies:
        greedy = []
        greedy_True = []
        
        if topology == 'lightning':
            greedy = lightning
            greedy_True = lightning_True
        elif topology == 'barabasi-albert':
            greedy = ba
            greedy_True = ba_True
        else:
            greedy = ws
            greedy_True = ws_True
        x = [number for number in range(1,11)]
        
        for element in greedy:
            if len(element) < 10:
                for index in range(len(element), 10):
                    element.append(element[-1])
        greedy = [list(a) for a in (zip(*greedy))]

        greedy_max = []
        greedy_min = []
        greedy_mean = []
        for element in greedy:
            greedy_mean.append(np.mean(element))
            greedy_max.append(1.96*np.std(element)/np.sqrt(10))
            greedy_min.append(1.96*np.std(element)/np.sqrt(10))
        
        greedy_err = [greedy_min, greedy_max]
        plt.errorbar(x, greedy_mean, yerr=greedy_err, label='Ignore Cycle Creation', lw=2)
        
        for element in greedy_True:
            if len(element) < 10:
                for index in range(len(element), 10):
                    element.append(element[-1])
        
        greedy_True = [list(a) for a in (zip(*greedy_True))]

        greedy_True_max = []
        greedy_True_min = []
        greedy_True_mean = []
        for element in greedy_True:
            greedy_True_mean.append(np.mean(element))
            greedy_True_max.append(1.96*np.std(element)/np.sqrt(10))
            greedy_True_min.append(1.96*np.std(element)/np.sqrt(10))
    
        greedy_True_err = [greedy_True_min, greedy_True_max]
        
        plt.errorbar(x, greedy_True_mean, yerr=greedy_True_err, label='Forces Cycle Creation', ls='dashed', lw=2)
        plt.legend(fontsize = 16)
        plt.ylabel('Incentive', fontsize=24)
        plt.xlabel('\# of Neighbors', fontsize = 24)
        
        plt.xticks(fontsize = 24)
        plt.yticks(fontsize = 24)
        
        plt.tight_layout()
        plt.grid()
        plt.savefig('../results/node_attachment_results/compare_cycles_'+str(topology)+str(alpha)+'.pdf', dpi=600)
        plt.clf()



def plot_bc_stats (alpha, cycle=False):
    topologies = ['lightning','barabasi-albert','watts-strogatz']
    algorithms = ['greedy', 'centrality', 'degree', 'rich', 'random']
    lightning = []
    ba = []
    ws = []
    node = 'new_node'
    os.chdir('../results/node_attachment_results/graphs')
    graph_files_all = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime)
    os.chdir('../../../src')
    
    graph_files = []
    for graph_file in graph_files_all:
        if graph_file.endswith(str(cycle)+'_alpha_'+str(alpha)+'.gml') and graph_file.startswith('lightning')==False:
            graph_files.append(graph_file)

    barabasi_graphs = []
    ws_graphs = []
    for graph_file in graph_files:
        if graph_file.startswith('barabasi'):
            barabasi_graphs.append(graph_file)
        if graph_file.startswith('watts-strogatz'):
            ws_graphs.append(graph_file)

    graph_files = graph_files[:20]

    for topology in topologies:
        for heuristic in algorithms:
            if topology != 'lightning':
                fp = open('../results/node_attachment_results/'+heuristic+'_'+str(cycle)+'_'+topology+'_alpha_'+str(alpha)+'.dat','rb')
            else:
                if heuristic == 'greedy':
                    fp = open('../results/node_attachment_results/'+heuristic+'_'+str(cycle)+'_'+topology+'_alpha_'+str(alpha)+'.dat','rb')
                else:
                    fp = open('../results/node_attachment_results/'+heuristic+'_False_'+topology+'_alpha_'+str(alpha)+'.dat','rb')

            while True:
                try:
                    (_, aux) = pickle.load(fp)
                    if topology == 'lightning':
                        lightning.append(aux)
                    if topology == 'barabasi-albert':
                        ba.append(aux)
                    if topology == 'watts-strogatz':
                        ws.append(aux)
                except EOFError:
                    break
    
    for topology in topologies:
        greedy = []
        centrality = []
        degree = []
        rich = []
        random = []
        counter = 0

        if topology == 'lightning':
            topology_vec = lightning
        elif topology == 'barabasi-albert':
            topology_vec = ba
        else:
            topology_vec = ws

        for element in topology_vec:
            if len(element) < 10:
                for index in range(len(element), 10):
                    element.append('')
            counter += 1
            if topology == 'lightning' and cycle == True: 
                if counter <= 1:
                    greedy.append(element)
                elif counter <= 11:
                    centrality.append(element)
                elif counter <= 21:
                    degree.append(element)
                elif counter <= 31:
                    rich.append(element)
                elif counter <= 41:
                    random.append(element)
            else:
                if counter <= 10:
                    greedy.append(element)
                elif counter <= 20:
                    centrality.append(element)
                elif counter <= 30:
                    degree.append(element)
                elif counter <= 40:
                    rich.append(element)
                elif counter <= 50:
                    random.append(element)
        
        if topology == 'lightning':
            Graph = graph_names('jul 2022')
            Graph = validate_graph(Graph)
            Graph = snowball_sample(Graph, size = 512)
            Graph.add_node(node)
            Graph = make_graph_payment(Graph, 4104693)

        width = 0.3
        r = np.arange(1)
        x = [i for i in range(1,11)]

        bc = []
        
        bc_mean = []
        bc_max = []
        bc_min = []
        
        counter = 0
        for channels in tqdm(greedy, desc='Calculating rewards for greedy algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            bc_in = []
            channel_counter = 0
            for edge in channels:
                if edge == '':
                    bc_in = bc[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                channel_counter += 1
                bc_in.append(nx.betweenness_centrality(graph_copy, normalized = True, weight='fee')[node])

            bc.append(bc_in)
        

        bc = [list(a) for a in (zip(*bc))]

        for element in bc:
            bc_mean.append(np.mean(element))
            bc_max.append(1.96*np.std(element)/np.sqrt(10))
            if np.mean(element) - 1.96*np.std(element)/np.sqrt(10) > 0:
                bc_min.append(1.96*np.std(element)/np.sqrt(10))
            else:
                bc_min.append(np.mean(element))
        bc_err = [bc_min, bc_max]
        #plt.bar(r, bc_mean, yerr=bc_err, label='ProfitPilot', width=width, color='#351431', edgecolor='k')
        plt.errorbar(x,bc_mean, yerr=bc_err, label='ProfitPilot', lw = 2)
        
        bc = []
        
        bc_mean = []
        bc_max = []
        bc_min = []
        
        counter = 0
        for channels in tqdm(centrality, desc='Calculating rewards for centrality algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            bc_in = []
            channel_counter = 0
            for edge in channels:
                if edge == '':
                    bc_in = bc[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                channel_counter += 1
                bc_in.append(nx.betweenness_centrality(graph_copy, normalized = True, weight='fee')[node])

            bc.append(bc_in)
        

        bc = [list(a) for a in (zip(*bc))]

        for element in bc:
            bc_mean.append(np.mean(element))
            bc_max.append(1.96*np.std(element)/np.sqrt(10))
            if np.mean(element) - 1.96*np.std(element)/np.sqrt(10) > 0:
                bc_min.append(1.96*np.std(element)/np.sqrt(10))
            else:
                bc_min.append(np.mean(element))
        bc_err = [bc_min, bc_max]
        #plt.bar(r + width, bc_mean, yerr=bc_err, label='Centrality', width=width, color='#775253', edgecolor='k')
        plt.errorbar(x,bc_mean, yerr=bc_err, label='Centrality', lw = 2, ls='dashed')

        bc = []
        
        bc_mean = []
        bc_max = []
        bc_min = []
        
        counter = 0
        for channels in tqdm(degree, desc='Calculating rewards for degree algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            bc_in = []
            channel_counter = 0
            for edge in channels:
                if edge == '':
                    bc_in = bc[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                channel_counter += 1
                bc_in.append(nx.betweenness_centrality(graph_copy, normalized = True, weight='fee')[node])

            bc.append(bc_in)
        

        bc = [list(a) for a in (zip(*bc))]

        for element in bc:
            bc_mean.append(np.mean(element))
            bc_max.append(1.96*np.std(element)/np.sqrt(10))
            if np.mean(element) - 1.96*np.std(element)/np.sqrt(10) > 0:
                bc_min.append(1.96*np.std(element)/np.sqrt(10))
            else:
                bc_min.append(np.mean(element))
        bc_err = [bc_min, bc_max]
        #plt.bar(r + 2*width, bc_mean, yerr=bc_err, label='Degree', width=width, color='#bdc696',edgecolor='k')
        plt.errorbar(x,bc_mean, yerr=bc_err, label='Degree', lw = 2, ls='dotted')

        bc = []
        
        bc_mean = []
        bc_max = []
        bc_min = []
        
        counter = 0
        for channels in tqdm(rich, desc='Calculating rewards for rich algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            bc_in = []
            channel_counter = 0
            for edge in channels:
                if edge == '':
                    bc_in = bc[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                channel_counter += 1
                bc_in.append(nx.betweenness_centrality(graph_copy, normalized = True, weight='fee')[node])

            bc.append(bc_in)
        

        bc = [list(a) for a in (zip(*bc))]

        for element in bc:
            bc_mean.append(np.mean(element))
            bc_max.append(1.96*np.std(element)/np.sqrt(10))
            if np.mean(element) - 1.96*np.std(element)/np.sqrt(10) > 0:
                bc_min.append(1.96*np.std(element)/np.sqrt(10))
            else:
                bc_min.append(np.mean(element))
        bc_err = [bc_min, bc_max]
        #plt.bar(r + 3*width, bc_mean, yerr=bc_err, label='Rich', width=width, color='#d1d3c4',edgecolor='k')
        plt.errorbar(x,bc_mean, yerr=bc_err, label='Rich', lw = 2, ls='dashdot')

        bc = []
        
        bc_mean = []
        bc_max = []
        bc_min = []
        
        counter = 0
        for channels in tqdm(random, desc='Calculating rewards for random algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            bc_in = []
            channel_counter = 0
            for edge in channels:
                if edge == '':
                    bc_in = bc[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                channel_counter += 1
                bc_in.append(nx.betweenness_centrality(graph_copy, normalized = True, weight='fee')[node])

            bc.append(bc_in)
        

        bc = [list(a) for a in (zip(*bc))]
        
        for element in bc:
            bc_mean.append(np.mean(element))
            bc_max.append(1.96*np.std(element)/np.sqrt(10))
            if np.mean(element) - 1.96*np.std(element)/np.sqrt(10) > 0:
                bc_min.append(1.96*np.std(element)/np.sqrt(10))
            else:
                bc_min.append(np.mean(element))
        bc_err = [bc_min, bc_max]
        #plt.bar(r + 4*width, bc_mean, yerr=bc_err, label='Random', width=width, color='#dfe0dc',edgecolor='k')
        plt.errorbar(x,bc_mean, yerr=bc_err, label='Random', lw = 2, ls=(0, (3, 1, 1, 1, 1, 1)))

        #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        #            mode="expand", borderaxespad=0, ncol=3, fontsize = 18, handlelength=1, handleheight=1)

        plt.legend(fontsize = 16)
        plt.grid()

        plt.ylabel('Probability of Collecting Fees',fontsize=24)
        plt.xlabel('\# of Neighbors',fontsize=24)
        
        #plt.tick_params(
        #    axis='x',          # changes apply to the x-axis
        #    which='both',      # both major and minor ticks are affected
        #    bottom=False,      # ticks along the bottom edge are off
        #    top=False,         # ticks along the top edge are off
        #    labelbottom=False)
        
        plt.yticks(fontsize=24)
        plt.xticks(fontsize=24)
        plt.tight_layout()
        
        plt.savefig('../results/node_attachment_results/line_collect_fee_'+topology+str(cycle)+'_alpha_'+str(alpha)+'.pdf', dpi=600)
        plt.clf()
    
def plot_cc_stats (alpha, cycle=False):
    topologies = ['lightning', 'barabasi-albert','watts-strogatz']
    algorithms = ['greedy', 'centrality', 'degree', 'rich', 'random']
    lightning = []
    ba = []
    ws = []
    node = 'new_node'
    os.chdir('../results/node_attachment_results/graphs')
    graph_files_all = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime)
    os.chdir('../../../src')

    graph_files = []
    for graph_file in graph_files_all:
        if graph_file.endswith(str(cycle)+'_alpha_'+str(alpha)+'.gml') and graph_file.startswith('lightning')==False:
            graph_files.append(graph_file)
    
    barabasi_graphs = []
    ws_graphs = []
    for graph_file in graph_files:
        if graph_file.startswith('barabasi'):
            barabasi_graphs.append(graph_file)
        if graph_file.startswith('watts'):
            ws_graphs.append(graph_file)


    graph_files = graph_files[:20]
    
    for topology in topologies:
        for heuristic in algorithms:
            if topology != 'lightning':
                fp = open('../results/node_attachment_results/'+heuristic+'_'+str(cycle)+'_'+topology+'_alpha_'+str(alpha)+'.dat','rb')
            else:
                if heuristic == 'greedy':
                    fp = open('../results/node_attachment_results/'+heuristic+'_'+str(cycle)+'_'+topology+'_alpha_'+str(alpha)+'.dat','rb')
                else:
                    fp = open('../results/node_attachment_results/'+heuristic+'_False_'+topology+'_alpha_'+str(alpha)+'.dat','rb')

            while True:
                try:
                    (_, aux) = pickle.load(fp)
                    if topology == 'lightning':
                        lightning.append(aux)
                    if topology == 'barabasi-albert':
                        ba.append(aux)
                    if topology == 'watts-strogatz':
                        ws.append(aux)
                except EOFError:
                    break
    
    for topology in topologies:
        greedy = []
        centrality = []
        degree = []
        rich = []
        random = []
        topology_vec = []
        counter = 0

        if topology == 'lightning':
            topology_vec = lightning
        elif topology == 'barabasi-albert':
            topology_vec = ba
        else:
            topology_vec = ws
        for element in topology_vec:
            if len(element) < 10:
                for index in range(len(element), 10):
                    element.append('')
            counter += 1
            if topology == 'lightning' and cycle == True: 
                if counter <= 1:
                    greedy.append(element)
                elif counter <= 11:
                    centrality.append(element)
                elif counter <= 21:
                    degree.append(element)
                elif counter <= 31:
                    rich.append(element)
                elif counter <= 41:
                    random.append(element)
            else:
                if counter <= 10:
                    greedy.append(element)
                elif counter <= 20:
                    centrality.append(element)
                elif counter <= 30:
                    degree.append(element)
                elif counter <= 40:
                    rich.append(element)
                elif counter <= 50:
                    random.append(element)
        if topology == 'lightning':
            Graph = graph_names('jul 2022')
            Graph = validate_graph(Graph)
            Graph = snowball_sample(Graph, size = 512)
            Graph.add_node(node)
            Graph = make_graph_payment(Graph, 4104693)

        width = 0.3
        r = np.arange(10)
        x = [i for i in range (1,11)]

        cc = []
        
        cc_mean = []
        cc_max = []
        cc_min = []
        
        counter = 0
        for channels in tqdm(greedy, desc='Calculating rewards for greedy algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            cc_in = []
            average_shortest_path = []
            for edge in channels:
                if edge == '':
                    cc_in = cc[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                p = nx.shortest_path_length(graph_copy, source=node, weight='fee')
                average_shortest_path = [p[element] for element in p]
                cc_in.append(np.mean(average_shortest_path))
                
            cc.append(cc_in)
        

        cc = [list(a) for a in (zip(*cc))]

        counter = 0
        for element in cc:
            counter += 1
            cc_mean.append(np.mean(element))
            cc_max.append(1.96*np.std(element)/np.sqrt(10))
            if np.mean(element) - 1.96*np.std(element)/np.sqrt(10) > 0:
                cc_min.append(1.96*np.std(element)/np.sqrt(10))
            else:
                cc_min.append(np.mean(element))
        cc_err = [cc_min, cc_max]
        #plt.bar(r, cc_mean, yerr=cc_err, label='ProfitPilot', width=width, color = '#545454',edgecolor='k')
        plt.errorbar(x, cc_mean, yerr=cc_err, label='ProfitPilot', lw = 2)

        cc = []
        
        cc_mean = []
        cc_max = []
        cc_min = []
        
        counter = 0
        for channels in tqdm(centrality, desc='Calculating rewards for centrality algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            cc_in = []
            for edge in channels:
                if edge == '':
                    cc_in = cc[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                p = nx.shortest_path_length(graph_copy, source=node, weight='fee')
                average_shortest_path = [p[element] for element in p]
                cc_in.append(np.mean(average_shortest_path))
            cc.append(cc_in)
        

        cc = [list(a) for a in (zip(*cc))]

        counter = 0
        for element in cc:
            counter += 1
            cc_mean.append(np.mean(element))
            cc_max.append(1.96*np.std(element)/np.sqrt(10))
            if np.mean(element) - 1.96*np.std(element)/np.sqrt(10) > 0:
                cc_min.append(1.96*np.std(element)/np.sqrt(10))
            else:
                cc_min.append(np.mean(element))
        cc_err = [cc_min, cc_max]
        #plt.bar(r + width, cc_mean, yerr=cc_err, label='Centrality', width=width, color='#69747c',edgecolor='k')
        plt.errorbar(x, cc_mean, yerr=cc_err, label='Centrality', ls = 'dashed', lw = 2)

        cc = []
        
        cc_mean = []
        cc_max = []
        cc_min = []
        
        counter = 0
        for channels in tqdm(degree, desc='Calculating rewards for degree algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            cc_in = []
            for edge in channels:
                if edge == '':
                    cc_in = cc[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                p = nx.shortest_path_length(graph_copy, source=node, weight='fee')
                average_shortest_path = [p[element] for element in p]
                cc_in.append(np.mean(average_shortest_path))

            cc.append(cc_in)

        cc = [list(a) for a in (zip(*cc))]

        counter = 0
        for element in cc:
            counter += 1
            cc_mean.append(np.mean(element))
            cc_max.append(1.96*np.std(element)/np.sqrt(10))
            if np.mean(element) - 1.96*np.std(element)/np.sqrt(10) > 0:
                cc_min.append(1.96*np.std(element)/np.sqrt(10))
            else:
                cc_min.append(np.mean(element))
        cc_err = [cc_min, cc_max]
 #       plt.bar(r + 2*width, cc_mean, yerr=cc_err, label='Degree', width=width, color='#6baa75',edgecolor='k')
        plt.errorbar(x, cc_mean, yerr=cc_err, label='Degree', lw = 2, ls='dotted')

        cc = []
        
        cc_mean = []
        cc_max = []
        cc_min = []
        
        counter = 0
        for channels in tqdm(rich, desc='Calculating rewards for rich algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            cc_in = []
            for edge in channels:
                if edge == '':
                    cc_in = cc[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                p = nx.shortest_path_length(graph_copy, source=node, weight='fee')
                average_shortest_path = [p[element] for element in p]
                cc_in.append(np.mean(average_shortest_path))

            cc.append(cc_in)

        cc = [list(a) for a in (zip(*cc))]

        counter = 0
        for element in cc:
            counter += 1
            cc_mean.append(np.mean(element))
            cc_max.append(1.96*np.std(element)/np.sqrt(10))
            if np.mean(element) - 1.96*np.std(element)/np.sqrt(10) > 0:
                cc_min.append(1.96*np.std(element)/np.sqrt(10))
            else:
                cc_min.append(np.mean(element))
        cc_err = [cc_min, cc_max]
 #       plt.bar(r + 3*width, cc_mean, yerr=cc_err, label='Rich', width=width, color='#84dd63',edgecolor='k')
        plt.errorbar(x, cc_mean, yerr=cc_err, label='Rich', lw = 2, ls='dashdot')

        cc = []
        
        cc_mean = []
        cc_max = []
        cc_min = []
        
        counter = 0
        for channels in tqdm(random, desc='Calculating rewards for random algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            cc_in = []
            for edge in channels:
                if edge == '':
                    cc_in = cc[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                p = nx.shortest_path_length(graph_copy, source=node, weight='fee')
                average_shortest_path = [p[element] for element in p]
                cc_in.append(np.mean(average_shortest_path))

            cc.append(cc_in)

        cc = [list(a) for a in (zip(*cc))]

        counter = 0
        for element in cc:
            counter += 1
            cc_mean.append(np.mean(element))
            cc_max.append(1.96*np.std(element)/np.sqrt(10))
            if np.mean(element) - 1.96*np.std(element)/np.sqrt(10) > 0:
                cc_min.append(1.96*np.std(element)/np.sqrt(10))
            else:
                cc_min.append(np.mean(element))
        cc_err = [cc_min, cc_max]
#        plt.bar(r + 4*width, cc_mean, yerr=cc_err, label='Random', width=width, color='#cbff4d',edgecolor='k')
        plt.errorbar(x, cc_mean, yerr=cc_err, label='Random', lw = 2, ls=(0, (3, 1, 1, 1, 1, 1)))

        #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        #            mode="expand", borderaxespad=0, ncol=3, fontsize = 18, handleheight = 1, handlelength = 1)
        plt.legend(fontsize = 16)

        plt.ylabel('Average Paid Fee (satoshis)',fontsize=24)
        plt.xlabel('\# of Neighbors', fontsize = 24)
        plt.grid()
        #plt.tick_params(
        #    axis='x',          # changes apply to the x-axis
        #    which='both',      # both major and minor ticks are affected
        #    bottom=False,      # ticks along the bottom edge are off
        #    top=False,         # ticks along the top edge are off
        #    labelbottom=False)
        
        plt.yticks(fontsize=24)
        plt.xticks(fontsize=24)
        plt.tight_layout()
        
        plt.savefig('../results/node_attachment_results/line_paid_fee_'+topology+str(cycle)+'_alpha_'+str(alpha)+'.pdf', dpi=600)
        plt.clf()
 
def number_triangles (alpha):
    cycle = True
    topologies = ['lightning', 'barabasi-albert','watts-strogatz']
    algorithms = ['greedy', 'centrality', 'degree', 'rich', 'random']
    lightning = []
    ba = []
    ws = []
    node = 'new_node'
    os.chdir('../results/node_attachment_results/graphs')
    graph_files_all = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime)
    os.chdir('../../../src')

    graph_files = []
    for graph_file in graph_files_all:
        if graph_file.endswith('True_alpha_'+str(alpha)+'.gml') and graph_file.startswith('lightning')==False:
            graph_files.append(graph_file)
    
    barabasi_graphs = []
    ws_graphs = []
    for graph_file in graph_files:
        if graph_file.startswith('barabasi'):
            barabasi_graphs.append(graph_file)
        if graph_file.startswith('watts'):
            ws_graphs.append(graph_file)


    graph_files = graph_files[:20]
    
    for topology in topologies:
        for heuristic in algorithms:
            if topology != 'lightning':
                fp = open('../results/node_attachment_results/'+heuristic+'_True_'+topology+'_alpha_'+str(alpha)+'.dat','rb')
            else:
                if heuristic == 'greedy':
                    fp = open('../results/node_attachment_results/'+heuristic+'_True_'+topology+'_alpha_'+str(alpha)+'.dat','rb')
                else:
                    fp = open('../results/node_attachment_results/'+heuristic+'_False_'+topology+'_alpha_'+str(alpha)+'.dat','rb')

            while True:
                try:
                    (_, aux) = pickle.load(fp)
                    if topology == 'lightning':
                        lightning.append(aux)
                    if topology == 'barabasi-albert':
                        ba.append(aux)
                    if topology == 'watts-strogatz':
                        ws.append(aux)
                except EOFError:
                    break
    
    for topology in topologies:
        greedy = []
        centrality = []
        degree = []
        rich = []
        random = []
        topology_vec = []
        counter = 0

        if topology == 'lightning':
            topology_vec = lightning
        elif topology == 'barabasi-albert':
            topology_vec = ba
        else:
            topology_vec = ws
        for element in topology_vec:
            if len(element) < 10:
                for index in range(len(element), 10):
                    element.append('')
            counter += 1
            if topology == 'lightning' and cycle == True: 
                if counter <= 1:
                    greedy.append(element)
                elif counter <= 11:
                    centrality.append(element)
                elif counter <= 21:
                    degree.append(element)
                elif counter <= 31:
                    rich.append(element)
                elif counter <= 41:
                    random.append(element)
            else:
                if counter <= 10:
                    greedy.append(element)
                elif counter <= 20:
                    centrality.append(element)
                elif counter <= 30:
                    degree.append(element)
                elif counter <= 40:
                    rich.append(element)
                elif counter <= 50:
                    random.append(element)
        if topology == 'lightning':
            Graph = graph_names('jul 2022')
            Graph = validate_graph(Graph)
            Graph = snowball_sample(Graph, size = 512)
            Graph.add_node(node)
            Graph = make_graph_payment(Graph, 4104693)

        width = 0.3
        r = np.arange(1)
        x = [i for i in range(1,11)]

        triangles = []
        cycles_heu = []
        cycles = []
        counter = 0
        for channels in tqdm(greedy, desc='Calculating rewards for greedy algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            triangles_edge = []
            average_shortest_path = []
            for edge in channels:
                if edge == '':
                    triangles_edge = triangles[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                triangles_edge.append(nx.triangles(graph_copy.to_undirected(),node))
                
            triangles.append(triangles_edge)
            for neighbor in graph_copy.neighbors(node):
                cycles.append(find_cycle(graph_copy, (node, neighbor), node, 4104693, length=True))
        cycles_heu.append(cycles)

        #plt.boxplot(cycles)
        #cycle_mean = []
        #for cycle in cycles_heu:
        #    cycle_mean.append(round(np.min(cycle)))
        #print('Mean:' + str(np.mean(cycle_mean)))
        #print('Error:'+str([1.96*np.std(cycle_mean)/np.sqrt(10)]))

        triangles = [list(a) for a in (zip(*triangles))]

        counter = 0
        triangle_mean = []
        triangle_max = []
        triangle_min = []
        for triangle in triangles:
            triangle_mean.append(np.mean(triangle))
            triangle_max.append(1.96*np.std(triangles[-1])/np.sqrt(10))
            if np.mean(triangle) - 1.96*np.std(triangle)/np.sqrt(10) > 0:
                triangle_min.append(1.96*np.std(triangle)/np.sqrt(10))
            else:
                triangle_min.append(np.mean(triangle))
        triangle_err = [triangle_min, triangle_max]
        #plt.bar(r, triangle_mean, yerr=triangle_err, label='ProfitPilot', width=width, color = '#0b132b',edgecolor='k')
        plt.errorbar(x, triangle_mean, yerr=triangle_err, label='ProfitPilot', lw = 2)


        triangles = []
        cycles = []
        counter = 0
        for channels in tqdm(centrality, desc='Calculating rewards for centrality algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            triangles_edge = []
            average_shortest_path = []
            for edge in channels:
                if edge == '':
                    triangles_edge = triangles[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                triangles_edge.append(nx.triangles(graph_copy.to_undirected(),node))
                
            triangles.append(triangles_edge)
            for neighbor in graph_copy.neighbors(node):
                cycles.append(find_cycle(graph_copy, (node, neighbor), node, 4104693, length=True))
        cycles_heu.append(cycles)
        
        #plt.boxplot(cycles)
        #cycle_mean = []
        #for cycle in cycles_heu:
        #    cycle_mean.append(round(np.min(cycle)))
        #print('Mean:' + str(np.mean(cycle_mean)))
        #print('Error:'+str([1.96*np.std(cycle_mean)/np.sqrt(10)]))

        triangles = [list(a) for a in (zip(*triangles))]

        triangle_mean = []
        triangle_max = []
        triangle_min = []
        for triangle in triangles:
            triangle_mean.append(np.mean(triangle))
            triangle_max.append(1.96*np.std(triangles[-1])/np.sqrt(10))
            if np.mean(triangle) - 1.96*np.std(triangle)/np.sqrt(10) > 0:
                triangle_min.append(1.96*np.std(triangle)/np.sqrt(10))
            else:
                triangle_min.append(np.mean(triangle))
        triangle_err = [triangle_min, triangle_max]
        #plt.bar(r+width, triangle_mean, yerr=triangle_err, label='Centrality', width=width, color = '#1c2541',edgecolor='k')
        plt.errorbar(x, triangle_mean, yerr=triangle_err, label='Centrality', lw = 2, ls='dashed')

        triangles = []
        cycles = []
        counter = 0
        for channels in tqdm(degree, desc='Calculating rewards for degree algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            triangles_edge = []
            average_shortest_path = []
            for edge in channels:
                if edge == '':
                    triangles_edge = triangles[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                triangles_edge.append(nx.triangles(graph_copy.to_undirected(),node))
                
            triangles.append(triangles_edge)
            for neighbor in graph_copy.neighbors(node):
                cycles.append(find_cycle(graph_copy, (node, neighbor), node, 4104693, length=True))
        cycles_heu.append(cycles)

        #plt.boxplot(cycles_heu)
        #cycle_mean = []
        #for cycle in cycles_heu:
        #    cycle_mean.append(round(np.min(cycle)))
        #print('Mean:' + str(np.mean(cycle_mean)))
        #print('Error:'+str([1.96*np.std(cycle_mean)/np.sqrt(10)]))
        triangles = [list(a) for a in (zip(*triangles))]

        counter = 0
        triangle_mean = []
        triangle_max = []
        triangle_min = []
        for triangle in triangles:
            triangle_mean.append(np.mean(triangle))
            triangle_max.append(1.96*np.std(triangles[-1])/np.sqrt(10))
            if np.mean(triangle) - 1.96*np.std(triangle)/np.sqrt(10) > 0:
                triangle_min.append(1.96*np.std(triangle)/np.sqrt(10))
            else:
                triangle_min.append(np.mean(triangle))
        triangle_err = [triangle_min, triangle_max]
        #plt.bar(r+2*width, triangle_mean, yerr=triangle_err, label='Degree', width=width, color = '#3a506b',edgecolor='k')
        plt.errorbar(x, triangle_mean, yerr=triangle_err, label='Degree', lw = 2, ls='dotted')

        triangles = []
        cycles = []
        counter = 0
        for channels in tqdm(rich, desc='Calculating rewards for rich algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            triangles_edge = []
            average_shortest_path = []
            for edge in channels:
                if edge == '':
                    triangles_edge = triangles[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                triangles_edge.append(nx.triangles(graph_copy.to_undirected(),node))
                
            triangles.append(triangles_edge)
            for neighbor in graph_copy.neighbors(node):
                cycles.append(find_cycle(graph_copy, (node, neighbor), node, 4104693, length=True))
        cycles_heu.append(cycles)
        

        #plt.boxplot(cycles_heu)
        #cycle_mean = []
        #for cycle in cycles_heu:
        #    cycle_mean.append(round(np.min(cycle)))
        #print('Mean:' + str(np.mean(cycle_mean)))
        #print('Error:'+str([1.96*np.std(cycle_mean)/np.sqrt(10)]))
        triangles = [list(a) for a in (zip(*triangles))]

        counter = 0
        triangle_mean = []
        triangle_max = []
        triangle_min = []
        for triangle in triangles:
            triangle_mean.append(np.mean(triangle))
            triangle_max.append(1.96*np.std(triangles[-1])/np.sqrt(10))
            if np.mean(triangle) - 1.96*np.std(triangle)/np.sqrt(10) > 0:
                triangle_min.append(1.96*np.std(triangle)/np.sqrt(10))
            else:
                triangle_min.append(np.mean(triangle))
        triangle_err = [triangle_min, triangle_max]
        #plt.bar(r+3*width, triangle_mean, yerr=triangle_err, label='Rich', width=width, color = '#5bc0be',edgecolor='k')
        plt.errorbar(x, triangle_mean, yerr=triangle_err, label='Rich', lw = 2, ls='dashdot')

        triangles = []
        cycles = []
        counter = 0
        for channels in tqdm(random, desc='Calculating rewards for random algorithm'):
            if topology != 'lightning':
                if topology == 'barabasi-albert':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+barabasi_graphs[counter])
                if topology == 'watts-strogatz':
                    Graph = nx.read_gml('../results/node_attachment_results/graphs/'+ws_graphs[counter])
                Graph = make_graph_payment(Graph, 4104693)
                counter += 1
                Graph.add_node(node)
            graph_copy = Graph.copy()
            triangles_edge = []
            average_shortest_path = []
            for edge in channels:
                if edge == '':
                    triangles_edge = triangles[-1]
                    continue
                graph_copy.add_edge(node, str(edge), fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                graph_copy.add_edge(str(edge), node, fee_base_msat = 100, fee_proportional_millionths = 50, fee = 305)
                triangles_edge.append(nx.triangles(graph_copy.to_undirected(),node))
                
            triangles.append(triangles_edge)
            for neighbor in graph_copy.neighbors(node):
                cycles.append(find_cycle(graph_copy, (node, neighbor), node, 4104693, length=True))
        cycles_heu.append(cycles)
        
        medianprops = dict(linewidth=2)
        #plt.boxplot(cycles_heu, labels=['ProfitPilot','Centrality','Degree','Rich','Random'], medianprops=medianprops, boxprops=medianprops)
        #cycle_mean = []
        #for cycle in cycles_heu:
        #    cycle_mean.append(round(np.min(cycle)))
        #print('Mean:' + str(np.mean(cycle_mean)))
        #print('Error:'+str([1.96*np.std(cycle_mean)/np.sqrt(10)]))


        triangles = [list(a) for a in (zip(*triangles))]

        counter = 0
        triangle_mean = []
        triangle_max = []
        triangle_min = []
        for triangle in triangles:
            triangle_mean.append(np.mean(triangle))
            triangle_max.append(1.96*np.std(triangles[-1])/np.sqrt(10))
            if np.mean(triangle) - 1.96*np.std(triangle)/np.sqrt(10) > 0:
                triangle_min.append(1.96*np.std(triangle)/np.sqrt(10))
            else:
                triangle_min.append(np.mean(triangle))
        triangle_err = [triangle_min, triangle_max]
        #plt.bar(r+4*width, triangle_mean, yerr=triangle_err, label='Random', width=width, color = '#ffffff',edgecolor='k')
        plt.errorbar(x, triangle_mean, yerr=triangle_err, label='Random', lw = 2, ls=(0, (3,1,1,1,1,1)))

        #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        #            mode="expand", borderaxespad=0, ncol=3, fontsize = 18, handleheight = 1, handlelength = 1)
        plt.legend(fontsize=16)
        plt.grid()

        #plt.ylabel('Rebalancing Fees',fontsize=24)
        plt.ylabel('\# of Triangles',fontsize=24)
        plt.xlabel('\# of Neighbors',fontsize=24)
        #plt.tick_params(
        #    axis='x',          # changes apply to the x-axis
        #    which='both',      # both major and minor ticks are affected
        #    bottom=False,      # ticks along the bottom edge are off
        #    top=False,         # ticks along the top edge are off
        #    labelbottom=False)
        
        plt.yticks(fontsize=24)
        plt.xticks(fontsize=24)
        plt.tight_layout()
        #plt.yscale('log')
        
        #plt.show()
        plt.savefig('../results/node_attachment_results/line_triangles_'+topology+'True_alpha_'+str(alpha)+'.pdf', dpi=600)
        plt.clf()
#number_triangles(0.5)
#plot_rewards(1.0, False)
#plot_bc_stats(0.5, False)
#compare_rewards_cycle()
