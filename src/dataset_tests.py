import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from topology import *
from pcn import *

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
    axin.annotate(text='38% of nodes have\n one neighbor only',xy=(2,0.358), xytext=(2.1,0.3563), arrowprops=dict(arrowstyle='->', lw=2), fontsize=14)
    plt.ylabel('CDF', fontsize=16)
    plt.xlabel('Node degree', fontsize=16)
    ax.tick_params(labelsize=16)
    axin.tick_params(labelsize=16)
    fig.savefig("../results/degree_distribution.pdf", dpi=300, bbox_inches='tight')

plt.style.use('seaborn-v0_8-colorblind')
Graph = graph_names('jul 2022')
Graph = validate_graph(Graph)
degree_distribution(Graph)
#check_ripple_seasonality()
#check_ripple_node_seasonality()
