import pandas as pd
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

import random
import statistics
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from scipy.fft import fft
from topology import *
from pcn import *
from cycle_finder import *
from tqdm import tqdm

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

#import tensorflow as tf
import tensorboard as tb

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from torch.autograd import Variable

from sklearn import preprocessing

plt.style.use('seaborn-v0_8-colorblind')
torch.set_num_threads(10)

class LSTM1(nn.Module):
    def __init__(self,num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
    def forward (self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

def visualize_time_series ():
    df = pd.read_csv('../datasets/transactions-in-USD-jan-2013-aug-2016.txt')

    df = df.loc[df['USD_amount'] < 10**(8)].reset_index()
    df = df.loc[df['USD_amount'] > 0].reset_index()
    df = df[(np.abs(stats.zscore(df['USD_amount'])) < 3)]
    df = df.sort_values(by=['unix_timestamp'], ascending=True).reset_index(drop=True)

    plt.plot(df['unix_timestamp'], df['USD_amount'])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Transaction Amount (\$)', fontsize = 20)
    plt.xlabel('Unix Time', fontsize = 20)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(18)
    ax.xaxis.get_offset_text().set_fontsize(18)
    plt.tight_layout()
    plt.savefig('../results/rebalancing-results/time-series.pdf', dpi=600)
    plt.clf()

    plot_acf(df['USD_amount'].tolist(), lags=50)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Autocorrelation', fontsize=20)
    plt.xlabel('Lag', fontsize=20)
    plt.tight_layout()
    plt.savefig('../results/rebalancing-results/autocorrelation.pdf',dpi=600)

def forecast_time_series_test ():
    print('Loading dataset...')
    df = pd.read_csv('../datasets/transactions-in-USD-jan-2013-aug-2016.txt')

    df = df.loc[df['USD_amount'] < 10**(8)].reset_index()
    df = df.loc[df['USD_amount'] > 0].reset_index()
    df = df[(np.abs(stats.zscore(df['USD_amount'])) < 3)]
    df['unix_timestamp'] = pd.to_datetime(df['unix_timestamp'], unit='s')
    df = df.loc[df['unix_timestamp'].dt.year >= 2016]
    df = df.loc[df['unix_timestamp'].dt.month > 5]
    df = df.sort_values(by=['unix_timestamp'], ascending=True).reset_index(drop=True)
    
    """As we want to predict transactions for the next day, we create a time index based on the day"""
    df['time_idx'] = df['unix_timestamp'].dt.year*365 + df['unix_timestamp'].dt.month*30 + df['unix_timestamp'].dt.day
    df['time_idx'] -= df['time_idx'].min()

    df['day'] = df.unix_timestamp.dt.day.astype(str).astype('category')

    max_prediction_length = 2
    max_encoder_length = 14
    training_cutoff = df['time_idx'].max() - max_prediction_length
    
    print('Creating training set...')

    training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx='time_idx',
            target='USD_amount',
            group_ids = ['sdr','rcv'],
            min_encoder_length = max_encoder_length // 2,
            max_encoder_length = max_encoder_length,
            static_categoricals = [],
            min_prediction_length = 1,
            max_prediction_length = max_prediction_length,
            time_varying_known_categoricals = ['day'],
            time_varying_known_reals = ['time_idx'],
            time_varying_unknown_categoricals = [],
            time_varying_unknown_reals = [
                'USD_amount',
            ],
            target_normalizer = None,
            add_relative_time_idx = True,
            add_target_scales = True,
            add_encoder_length = True,
            allow_missing_timesteps = True,
        )
    
    """Creating validation set to predict the last max_prediction_length days of the dataset"""
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    """Create dataloaders"""
    print('Creating daloaders...')
    batch_size = 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=5)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=5)

    #baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
    #MAE()(baseline_predictions.output, baseline_predictions.y)

    pl.seed_everything(42)
    
    print('Training the TFT model...')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience = 10, verbose = False, mode='min')
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger('lightning_logs')

    trainer = pl.Trainer(
            max_epochs = 50,
            accelerator = 'cpu',
            enable_model_summary = True,
            gradient_clip_val = 0.1,
            limit_train_batches = 50,
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate = 0.03,
            hidden_size = 2,
            attention_head_size = 2,
            dropout = 0.2,
            hidden_continuous_size = 2,
            loss = QuantileLoss(),
            log_interval = 0,
            optimizer = 'Ranger',
            reduce_on_plateau_patience = 4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator='cpu'))
    MAE()(predictions.output, predictions.y)

def create_dataset(dataset, lookback):
    x, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        x.append(feature)
        y.append(target)
    return torch.tensor(x), torch.tensor(y)

def forecast_timeseries_lstm():
    df = pd.read_csv('../datasets/transactions-in-USD-jan-2013-aug-2016.txt')

    df = df.loc[df['USD_amount'] < 10**(8)].reset_index(drop=True)
    df = df.loc[df['USD_amount'] > 0].reset_index(drop=True)
    df = df[(np.abs(stats.zscore(df['USD_amount'])) < 3)]
    df['unix_timestamp'] = pd.to_datetime(df['unix_timestamp'], unit='s')
    #df = df.loc[df['unix_timestamp'].dt.year >= 2014]
    #df = df.loc[df['unix_timestamp'].dt.month > 7]
    #df = df.loc[df['unix_timestamp'].dt.day > 14]
    df = df.sort_values(by=['unix_timestamp'], ascending=True).reset_index(drop=True)

    df = df[[c for c in df if c not in ['USD_amount']] + ['USD_amount']]

    x = df.iloc[:, :-1]
    y = df.iloc[:, 4:5]

    mm = preprocessing.MinMaxScaler()
    #mm = preprocessing.RobustScaler()
    x_le = x.apply(preprocessing.LabelEncoder().fit_transform)
    y_mm = mm.fit_transform(y)

    train_size = int(len(df['USD_amount'])*0.67)
    test_size = len(df['USD_amount']) - train_size

    x_train = x_le.iloc[:train_size, :]
    x_test = x_le.iloc[train_size:, :]

    y_train = y_mm[:train_size, :]
    y_test = y_mm[train_size:, :]

    print('Training shape', x_train.shape, y_train.shape)
    print('Test shape', x_test.shape, y_test.shape)

    lookback = 4

    x_train_tensors = Variable(torch.Tensor(x_train.values))
    x_test_tensors = Variable(torch.Tensor(x_test.values))
    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    x_train_tensors_final = torch.reshape(x_train_tensors, (x_train_tensors.shape[0], 1, x_train_tensors.shape[1]))
    x_test_tensors_final = torch.reshape(x_test_tensors, (x_test_tensors.shape[0], 1, x_test_tensors.shape[1]))

    print('Training shape', x_train_tensors_final.shape, y_train_tensors.shape)
    print('Test shape', x_test_tensors_final.shape, y_test_tensors.shape)

    model = LSTM1(num_classes=1, input_size = lookback, hidden_size = 10, num_layers = 1, seq_length = x_train_tensors_final.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    #loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle=True, batch_size=64, num_workers = 10, prefetch_factor = 20)

    loss_vec = []
    n_epochs = 200
    for epoch in range(n_epochs):
        outputs = model.forward(x_train_tensors_final)
        optimizer.zero_grad()

        loss = loss_fn(outputs, y_train_tensors)

        loss.backward()

        optimizer.step()
        if epoch % 1 == 0:
            print('Epoch: %d, loss: %1.5f' % (epoch, loss.item()))
            loss_vec.append(loss.item())

    df_x_le = df.iloc[:, :-1].apply(preprocessing.LabelEncoder().fit_transform)
    df_y_mm = mm.transform(df.iloc[:, -1:])

    df_x_le = Variable(torch.Tensor(df_x_le.values))
    df_y_mm = Variable(torch.Tensor(df_y_mm))

    df_x_le = torch.reshape(df_x_le, (df_x_le.shape[0],1,df_x_le.shape[1]))

    train_predict = model(df_x_le)
    data_predict = train_predict.data.numpy()
    datay_plot = df_y_mm.data.numpy()

    data_predict = mm.inverse_transform(data_predict)
    datay_plot = mm.inverse_transform(datay_plot)

    plt.plot(datay_plot, label='Actual Data')
    plt.plot(data_predict, label='Predicted Data')
    plt.axvline(x=train_size, c='g', ls='--')
    plt.legend()
    plt.savefig('../results/lstm.pdf', dpi=600, bbox_inches='tight')

    plt.clf()
    plt.plot(loss_vec)
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epochs', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('../results/loss_lstm.pdf', dpi=600, bbox_inches='tight')

forecast_timeseries_lstm()



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

###########################################################################################################################
#               Node attachment tests below
##########################################################################################################################
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
    snapshots = [snapshots[1], snapshots[3]]
    print(snapshots)
    graphs = []
    for snapshot in tqdm(snapshots, desc='Evaluating snapshots'):
        graphs.append(nx.read_gml('../ln-snapshots/'+snapshot))
    intersection_graph = nx.intersection_all(graphs)
    edge_base_fees = {}
    edge_rate_fees = {}
    for (i,j) in intersection_graph.edges():
        base = graphs[0][i][j]['fee_base_msat']
        rate = graphs[0][i][j]['fee_proportional_millionths']
        edge_base_fees[(i,j)] = (base,True)
        edge_rate_fees[(i,j)] = (rate,True)
    
    for edge in edge_base_fees:
        (i,j) = edge
        (base, same_base) = edge_base_fees[(i,j)]
        (rate, same_rate) = edge_rate_fees[(i,j)]
        if graphs[1][i][j]['fee_base_msat'] != base:  
            edge_base_fees[(i,j)] = (base,False)
        if graphs[1][i][j]['fee_proportional_millionths'] != rate:
            edge_rate_fees[(i,j)] = (rate,False)

    rate_counter = 0
    base_counter = 0
    base_difference = []
    rate_difference = []
    for (i,j) in edge_base_fees:
        (base, same) = edge_base_fees[(i,j)]
        if same == True:
            base_counter += 1
        else:
            base_difference.append((base,graphs[1][i][j]['fee_base_msat']))
        (rate, same) = edge_rate_fees[(i,j)]
        if same == True:
            rate_counter += 1
        else:
            rate_difference.append((rate,graphs[1][i][j]['fee_proportional_millionths']))
    print('Base fee didn\'t change: ' + str(base_counter/len(edge_base_fees)))
    print('Rate fee didn\'t change: ' + str(rate_counter/len(edge_rate_fees)))

    decreased_base = 0
    zero_counter = 0
    for (before,after) in rate_difference:
        if after < before:
            decreased_base += 1
        if after == 0:
            zero_counter += 1
    print('Decreased base fee: ' + str(decreased_base/len(base_difference)))
    print('Removed base fee: ' + str(zero_counter/len(base_difference)))

    cycle_cost = []
    for node in tqdm(intersection_graph.nodes(), desc='Checking cheapest cycle change'):
        graph0_copy = graphs[0].copy()
        graph1_copy = graphs[1].copy()
        for neighbor in graphs[0].neighbors(node):
            if neighbor not in graphs[1].neighbors(node):
                continue
            attr_ij = graph_copy[node][neighbor]
            attr_ji = graph_copy[neighbor][node]
                
            graph0_copy.remove_edge(node, neighbor)
            graph0_copy.remove_edge(neighbor, node)
            graph1_copy.remove_edge(node, neighbor)
            graph1_copy.remove_edge(neighbor, node)

            if nx.has_path(graph0_copy, node, neighbor) and nx.has_path(graph1_copy, node, neighbor):    
                cheapest_cycle[(node,neighbor)] = nx.shortest_path(graph_copy, source=node, target=neighbor, weight='fee', method='dijkstra')
                shortest_cycle[(node,neighbor)] = nx.shortest_path(graph_copy, source=node, target=neighbor, method='dijkstra')
            else:
                no_cycle += 1

            graph_copy.add_edge(node, neighbor)
            graph_copy.add_edge(neighbor, node)
            graph_copy[node][neighbor].update(attr_ij)
            graph_copy[neighbor][node].update(attr_ji)


#check_fees_change()
#plt.style.use('seaborn-v0_8-colorblind')
#Graph = graph_names('jul 2022')
#Graph = validate_graph(Graph)
#cheapest_cycles_vs_shortest_cycles(Graph)
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
