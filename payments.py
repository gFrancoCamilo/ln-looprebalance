from pcn import *
from topology import *
import pandas as pd
import numpy as np
from scipy import stats

ONE_DOLLAR_IN_SATOSHI = 4968

def get_payment_dataset (option: str = 'ripple'):
    if option == 'ripple':
        payments = pd.read_csv('datasets/transactions-in-USD-jan-2013-aug-2016.txt')
        payments = payments['USD_amount']
        payments = filter_data(payments)
    elif option == 'credit-card':
        payments = pd.read_csv('datasets/creditcard.csv')
        payments = payments['Amount']
        payments = filter_data(payments)
    else:
        raise Exception ('Invalid payment option selected')
    return payments

def filter_data (payments: pd.DataFrame):
    payments = payments.loc[payments < 10**(8)]
    payments = payments.loc[payments > 0]
    payments = payments[(np.abs(stats.zscore(payments)) < 3)]
    return payments

def get_end_hosts (Graph: nx.DiGraph, n: int = 4):
    end_hosts = []
    desc = 'Selecting end hosts to issue payments'
    for node in tqdm(Graph.nodes(), desc=desc):
        if Graph.degree(node) <= n:
            end_hosts.append(node)
    return end_hosts

def choose_payments (payment_data: pd.DataFrame, n_payments: int = 10):
    payments = 0
    payment_values = []
    
    while payments < n_payments:
        payment_values.append(np.random.choice(payment_data.to_list()))
        payments += 1
    payments_satoshis = convert_dolars_to_satoshi(payment_values)
    return payments_satoshis

def convert_dolars_to_satoshi (payments: list) -> list:
    desc = 'Converting from dolar to satoshi using exchange rate 1->' + str(ONE_DOLLAR_IN_SATOSHI)
    for index in tqdm(range(len(payments)), desc=desc):
        payments[index] = round(payments[index] * ONE_DOLLAR_IN_SATOSHI)
    return payments

def get_payments_ln (Graph: nx.DiGraph, list_payments: list, n: int = 4):
    end_hosts = get_end_hosts(Graph, n)
    payments = 0
    payments_dict = {}
    while payments < len(list_payments):
        source = np.random.choice(end_hosts)
        destination = np.random.choice(end_hosts)
        if source != destination:
            payments_dict[(source,destination)] = list_payments[payments]
            payments += 1
    return payments_dict
    