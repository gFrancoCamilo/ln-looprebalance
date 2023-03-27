from pcn import *
from topology import *
import pandas as pd
import numpy as np
from scipy import stats

ONE_DOLLAR_IN_SATOSHI = 4968

def get_payment_dataset (option: str = 'ripple'):
    """
    get_payment dataset reads datasets and transforms them in pandas DataFrame format.
    It receives the option chosen by the user as arguments. Currently, we offer two options:
    ripple and credit-card. As the name suggests, if the selected option is ripple, the function
    returns the loaded ripple dataset. Otherwise, if credit-card is selected, the function returns
    the laoded credit-card dataset.
    """
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
    """
    filter_data filters payment data from dataset. Specifically, it removes outliers,
    such as payments above 100M dolars and payments equal to 0 dolars. Furthermore,
    it removes every payment with over 3 standard deviations from the average.
    """
    payments = payments.loc[payments < 10**(8)]
    payments = payments.loc[payments > 0]
    payments = payments[(np.abs(stats.zscore(payments)) < 3)]
    return payments

def get_end_hosts (Graph: nx.DiGraph, n: int = 7):
    """
    get_end_hosts selects possible end-hosts to issue payments in the network.
    As most of the payments in PCNs come from low-degree nodes, we get nodes that
    present degree lower than 4 and return them to be selected as possible senders
    or receiver of payments.
    """
    end_hosts = []
    desc = 'Selecting end hosts to issue payments'
    for node in tqdm(Graph.nodes(), desc=desc):
        if Graph.degree(node) <= n:
            end_hosts.append(node)
    return end_hosts

def choose_payments (payment_data: pd.DataFrame, n_payments: int = 10):
    """
    choose_payments selects n random payments from a given dataset to be used in the
    simulation. It receives as arguments the dataset in pandas DataFrame format and the
    number of payments it has to select. The function returns the randomly selected payment
    values.
    """
    payments = 0
    payment_values = []
    
    while payments < n_payments:
        payment_values.append(np.random.choice(payment_data.to_list()))
        payments += 1
    payments_satoshis = convert_dolars_to_satoshi(payment_values)
    return payments_satoshis

def convert_dolars_to_satoshi (payments: list) -> list:
    """
    convert_dolars_to_satoshi converts dolar payments to satoshi payments. The
    function uses a fixed conversionn rate defined in the macro ONE_DOLLAR_IN_SATOSHI.
    """
    desc = 'Converting from dolar to satoshi using exchange rate 1->' + str(ONE_DOLLAR_IN_SATOSHI)
    for index in tqdm(range(len(payments)), desc=desc):
        payments[index] = round(payments[index] * ONE_DOLLAR_IN_SATOSHI)
    return payments

def get_payments_ln (Graph: nx.DiGraph, list_payments: list, n: int = 7):
    """
    get_payments_ln selects a random pair of end-hosts to act as sender-receiver
    and associates it with a payment to issui in the simulation. The function receives
    the network Graph, the list of payments and the threshold node degree to consider a 
    node as an end-host or not. The function returns a dictionary with the pair (sender, receiver)
    as key and the selected payment value as value.
    """
    end_hosts = get_end_hosts(Graph, n)
    payments = 0
    payments_dict = {}
    print ("Attributing selected payments to selected end-hosts. This may take a while...")
    while payments < len(list_payments):
        source = np.random.choice(end_hosts)
        destination = np.random.choice(end_hosts)
        source_node_balance = get_node_balance(Graph, source)
        if source_node_balance > list_payments[payments] and source != destination:
            payments_dict[(source,destination)] = list_payments[payments]
            payments += 1
    return payments_dict
    