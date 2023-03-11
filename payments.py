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
    for node in Graph.nodes():
        if Graph.degree(node) <= 4:
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
    for index in range(len(payments)):
        payments[index] = round(payments[index] * ONE_DOLLAR_IN_SATOSHI)
    return payments