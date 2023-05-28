from topology import *
from pcn import *
from payments import *
from cycle_finder import *

import os
import pickle

import networkx as nx
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

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

def plot_rewards ():
    topologies = ['lightning','barabasi-albert','watts-strogatz']
    algorithms = ['greedy', 'centrality', 'degree', 'rich', 'random']
    
    lightning = []
    ba = []
    ws = []

    for topology in topologies:
        for heuristic in algorithms:
            fp = open('../results/node_attachment_results/'+heuristic+'_False_'+topology+'.dat','rb')
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
    plt.errorbar(x, greedy_mean, yerr=greedy_err, label='Greedy', lw=2)

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
    plt.legend()

    plt.ylabel('Reward', fontsize=16)
    plt.xlabel('Channels Created', fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    plt.tight_layout()
    plt.grid()
    plt.savefig('../results/node_attachment_results/lightning_False.pdf', dpi=600)
    plt.clf()

    greedy = []
    centrality = []
    degree = []
    rich = []
    random = []
    counter = 0
    for element in ba:
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
    plt.errorbar(x, greedy_mean, yerr=greedy_err, label='Greedy', lw=2)

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
    plt.legend()

    plt.ylabel('Reward', fontsize=16)
    plt.xlabel('Channels Created', fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    plt.tight_layout()
    plt.grid()
    plt.savefig('../results/node_attachment_results/ba_False.pdf', dpi=600)
    plt.clf()

    greedy = []
    centrality = []
    degree = []
    rich = []
    random = []
    counter = 0
    for element in ws:
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
    plt.errorbar(x, greedy_mean, yerr=greedy_err, label='Greedy', lw=2)

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
    plt.legend()

    plt.ylabel('Reward', fontsize=16)
    plt.xlabel('Channels Created', fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    plt.tight_layout()
    plt.grid()
    plt.savefig('../results/node_attachment_results/ws_False.pdf', dpi=600)

plt.style.use('seaborn-v0_8-colorblind')
plot_rewards()
