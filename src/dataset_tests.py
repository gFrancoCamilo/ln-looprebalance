import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import statistics
import os
import pickle
from scipy.fft import fft
from topology import *
from pcn import *
from cycle_finder import *
from tqdm import tqdm

def check_ripple_seasonality ():
    df = pd.read_csv('../datasets/transactions-in-USD-jan-2013-aug-2016.txt')

    df = df.loc[df['USD_amount'] < 10**(8)].reset_index()
    df = df.loc[df['USD_amount'] > 0].reset_index()
    df = df.sort_values(by=['unix_timestamp'], ascending=True).reset_index(drop=True)

    initial_timestamp = 1388534400
    timestamp_iterator = 1388534400
    transactions_day = []
    day_accumulator = 0
    index = 0
    while timestamp_iterator < (initial_timestamp + 2*365*24*60*60):
        while df['unix_timestamp'][index] < initial_timestamp:
            index += 1
        while df['unix_timestamp'][index] < timestamp_iterator + (24*60*60):
            day_accumulator += df['USD_amount'][index]
            index += 1
        transactions_day.append(day_accumulator)
        timestamp_iterator = timestamp_iterator + (24*60*60)
        day_accumulator = 0
    
    transactions_week = []
    week_accumulator = 0
    index = 0
    for day in transactions_day:
        if index == 7:
            transactions_week.append(week_accumulator/index)
            week_accumulator = 0
            index = 0
        week_accumulator += day
        index += 1      

    plt.figure()
    plt.plot(transactions_day, lw=1.5)
    plt.ylabel('Valor das Transações (US$)', fontsize=16)
    plt.xlabel('Dia', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(16)
    plt.tight_layout()
    plt.savefig('../results/transactions_day.png', dpi=600)

    plt.clf()
    plt.plot(transactions_week, lw=1.5)
    plt.ylabel('Valor das Transações (US$)', fontsize=16)
    plt.xlabel('Semana', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(16)
    plt.tight_layout()
    plt.savefig('../results/transactions_week.png', dpi=600)

    x = np.abs(fft(transactions_day))
    plt.clf()
    plt.plot(x, lw=1.5)
    plt.ylabel('Valor Absoluto da Componente', fontsize=16)
    plt.xlabel('Peridiocidade em Dias',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(16)
    plt.tight_layout()
    plt.savefig('../results/fft_day.png', dpi=600)

    x = np.abs(fft(transactions_week))
    plt.clf()
    plt.plot(x, lw=1.5)
    plt.ylabel('Valor Absoluto da Componente', fontsize=16)
    plt.xlabel('Peridiocidade em Semanas',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(16)
    plt.tight_layout()
    plt.savefig('../results/fft_week.png', dpi=600)


def check_ripple_node_seasonality ():
    df = pd.read_csv('../datasets/transactions-in-USD-jan-2013-aug-2016.txt')

    df = df.loc[df['USD_amount'] < 10**(8)].reset_index()
    df = df.loc[df['USD_amount'] > 0].reset_index()
    df = df.sort_values(by=['unix_timestamp'], ascending=True).reset_index(drop=True)

    node_sends_most1 = 'rMQ19FYnZaSXjpd5bZrTJYfc4VSQr6cNcp'
    
    sdr = df.loc[df['sdr'] == node_sends_most1]
    sdr = sdr.sort_values(by='unix_timestamp').reset_index(drop=True)

    initial_timestamp = 1396938800
    timestamp_iterator = 1396938800
    transactions_day = []
    day_accumulator = 0
    index = 0
    while timestamp_iterator < (initial_timestamp + 365*24*60*60):
        while sdr['unix_timestamp'][index] < initial_timestamp:
            index += 1
        while sdr['unix_timestamp'][index] < timestamp_iterator + (24*60*60):
            day_accumulator += df['USD_amount'][index]
            index += 1
        transactions_day.append(day_accumulator)
        timestamp_iterator = timestamp_iterator + (24*60*60)
        day_accumulator = 0
    
    transactions_week = []
    week_accumulator = 0
    index = 0
    for day in transactions_day:
        if index == 7:
            transactions_week.append(week_accumulator/index)
            week_accumulator = 0
            index = 0
        week_accumulator += day
        index += 1      

    plt.figure()
    plt.plot(transactions_day, lw=1.5)
    plt.ylabel('Valor das Transações (US$)', fontsize=16)
    plt.xlabel('Dia', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(16)
    plt.tight_layout()
    plt.savefig('../results/transactions_day_node.pdf', dpi=600)

    plt.clf()
    plt.plot(transactions_week, lw=1.5)
    plt.ylabel('Valor das Transações (US$)', fontsize=16)
    plt.xlabel('Semana', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(16)
    plt.tight_layout()
    plt.savefig('../results/transactions_week_node.pdf', dpi=600)

    x = np.abs(fft(transactions_day))
    plt.clf()
    plt.plot(x, lw=1.5)
    plt.ylabel('Valor Absoluto da Componente', fontsize=16)
    plt.xlabel('Peridiocidade em Dias',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(16)
    plt.tight_layout()
    plt.savefig('../results/fft_day_node.pdf', dpi=600)

    x = np.abs(fft(transactions_week))
    plt.clf()
    plt.plot(x, lw=1.5)
    plt.ylabel('Valor Absoluto da Componente', fontsize=16)
    plt.xlabel('Peridiocidade em Semanas',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(16)
    plt.tight_layout()
    plt.savefig('../results/fft_week_node.pdf', dpi=600)

def degree_distribution(Graph: nx.DiGraph):
    degrees = [Graph.degree(node) for node in Graph.nodes()]
    fig = plt.figure()
    plt.hist(degrees, density=True, cumulative=True, histtype='step', bins=5000, linewidth=2)
    ax = plt.gca()
    ax.set_xlim(-200,2791.8)
    axin = ax.inset_axes([0.4,0.15,0.5,0.4])
    axin.hist(degrees,density=True, cumulative=True, histtype='step', bins=5000, linewidth=2)
    axin.set_xlim(1,5.5)
    axin.set_ylim(0.3560,0.3595)
    ax.indicate_inset_zoom(axin, edgecolor='k')
    axin.grid()
    axin.annotate(text=r'$\approx$36% of nodes have'+ '\none neighbor only',xy=(2,0.358), xytext=(2.1,0.3563), arrowprops=dict(arrowstyle='->', lw=2), fontsize=14)
    plt.ylabel('CDF', fontsize=16)
    plt.xlabel('Node degree', fontsize=16)
    ax.tick_params(labelsize=16)
    axin.tick_params(labelsize=16)
    plt.show()
    fig.savefig("../results/degree_distribution.pdf", dpi=300, bbox_inches='tight')

def check_cycles (Graph: nx.DiGraph, degree_check = 4):
    degrees = list(Graph.degree())
    nodes_with_degree_two = []
    for (node, degree) in degrees:
        if degree == degree_check:
            nodes_with_degree_two.append(node)
    
    cycles = []
    for node in tqdm(nodes_with_degree_two, desc='Checking for cycles'):
        neighbors = [n for n in Graph[node]]
        for neighbor in neighbors:
            try:
                cycles.append(find_cycle (Graph, (node, neighbor), node, 4104693))
            except:
                cycles.append([])

    no_cycle = 0
    for cycle in cycles:
        if len(cycle) == 0:
            no_cycle += 1
    result = no_cycle/len(cycles)
    return (len(cycles),result)

def check_cycles_cost (Graph: nx.DiGraph, degree_check=4):
    degrees = list(Graph.degree())
    nodes_with_degree_two = []
    for (node, degree) in degrees:
        if degree == degree_check:
            nodes_with_degree_two.append(node)
    
    cycles = []
    for node in tqdm(nodes_with_degree_two, desc='Checking for cycles'):
        neighbors = [n for n in Graph[node]]
        for neighbor in neighbors:
            try:
                cycles.append(find_cycle(Graph, (node, neighbor), node, 4104693, length=True))
            except:
                continue

    return cycles

def plot_cycles_cost ():
    cost_file = open('../results/check_cycles_cost.dat','rb')
    y = []
    while True:
        try:
            (value, cost) = pickle.load(cost_file)
            y.append(cost)
        except EOFError:
            break
    medianprops = dict(linewidth=2)
    fig = plt.figure()
    plt.boxplot(y, labels=['4',' ','8',' ','12',' ','16',' ','20',' ','24',' ','28',' ','32',' ','36',' ','40'], medianprops=medianprops, boxprops=medianprops)
    plt.ylabel('Rebalancing Cost (in satoshis)', fontsize=16)
    plt.yscale('log')
    plt.xlabel('Node Degree', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    fig.savefig("../results/cycles_cost.pdf", dpi=300, bbox_inches='tight')

def cheapest_cycles_vs_shortest_cycles (Graph: nx.DiGraph):
    cheapest_cycle = {}
    shortest_cycle = {}
    no_cycle = 0
    equal = 0
    total = 0

    payment_graph = make_graph_payment(Graph.copy(), 4104693)
    graph_copy = payment_graph.copy()

    for node in tqdm(Graph.nodes(), desc='Checking shortest and cheapest cycles'):
        for neighbor in Graph.neighbors(node):
            attr_ij = graph_copy[node][neighbor]
            attr_ji = graph_copy[neighbor][node]
                
            graph_copy.remove_edge(node, neighbor)
            graph_copy.remove_edge(neighbor, node)

            if nx.has_path(graph_copy, node, neighbor):    
                cheapest_cycle[(node,neighbor)] = nx.shortest_path(graph_copy, source=node, target=neighbor, weight='fee', method='dijkstra')
                shortest_cycle[(node,neighbor)] = nx.shortest_path(graph_copy, source=node, target=neighbor, method='dijkstra')
            else:
                no_cycle += 1

            graph_copy.add_edge(node, neighbor)
            graph_copy.add_edge(neighbor, node)
            graph_copy[node][neighbor].update(attr_ij)
            graph_copy[neighbor][node].update(attr_ji)

    increase = []
    for (i,j) in cheapest_cycle:
        if cheapest_cycle[(i,j)] == shortest_cycle[(i,j)]:
            equal += 1
        else:
            cheapest_accumulator = 0
            for index in range(len(cheapest_cycle[(i,j)])-1):
                cheapest_accumulator += graph_copy[cheapest_cycle[(i,j)][index]][cheapest_cycle[(i,j)][index+1]]['fee']
            shortest_accumulator = 0
            for index in range(len(shortest_cycle[(i,j)])-1):
                shortest_accumulator += graph_copy[shortest_cycle[(i,j)][index]][shortest_cycle[(i,j)][index+1]]['fee']
            increase.append((cheapest_accumulator,shortest_accumulator))
    difference = []
    for (cheapest, shortest) in increase:
        if cheapest != 0:
            difference.append(shortest/cheapest)

    print('No cycle: ' + str(no_cycle))
    print('Length: ' + str(len(cheapest_cycle)))
    print('Average Difference: ' + str(np.mean(difference)))
    print(equal/len(cheapest_cycle))

def check_fees_change ():
    snapshots = sorted([f for f in os.listdir('../ln-snapshots/') if f.endswith('recovered-capacitated.gml')])
    snapshots = [snapshots[0], snapshots[2]]
    print(snapshots)
    graphs = []
    for snapshot in tqdm(snapshots, desc='Evaluating snapshots'):
        graphs.append(nx.read_gml('../ln-snapshots/'+snapshot))
    for graph in graphs:
        print(graph.number_of_edges())
    intersection_graph = nx.intersection_all(graphs)
    print(len(intersection_graph.edges()))
    edge_fees = {}
    for (i,j) in intersection_graph.edges():
        base = graphs[0][i][j]['fee_base_msat']
        rate = graphs[0][i][j]['fee_proportional_millionths']
        edge_fees[(i,j)] = ((base,rate),True)
    
    for graph in graphs:
        for edge in edge_fees:
            (i,j) = edge
            (fees, same) = edge_fees[(i,j)]
            (base, rate) = fees
            if graph[i][j]['fee_base_msat'] != base or graph[i][j]['fee_proportional_millionths'] != rate:
                edge_fees[(i,j)] = (fees,False)
    counter = 0
    for (i,j) in edge_fees:
        (_, same) = edge_fees[(i,j)]
        if same == True:
            counter += 1
    print(counter/len(edge_fees))


#check_fees_change()
plt.style.use('seaborn-v0_8-colorblind')
Graph = graph_names('jul 2022')
Graph = validate_graph(Graph)
cheapest_cycles_vs_shortest_cycles(Graph)
#degree_distribution(Graph)

#my_file = open("../results/check_cycles_cost.dat", "wb")
#for i in range(4,42,2):
#    try:
#        cycles = check_cycles_cost(Graph, degree_check = i)
#        to_file = (i,cycles)
#        pickle.dump(to_file, my_file)
        #my_file.write(str(len_cycles) + ',' + str(result) + '\n')
#    except Exception as e:
#        error_file = open('../results/error.txt','w')
#        error_file.write(str(e))
#        error_file.close()
        
#my_file.close()
#file = open("../results/check_cycles_cost.txt", "w")
#for i in range(4,22,2):
#    result = check_cycles_cost(Graph)
#    file.write(result + '\n')
#file.close()

#plot_cycles_cost()

#check_ripple_seasonality()
#check_ripple_node_seasonality()
