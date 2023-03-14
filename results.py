from topology import *
from pcn import *
from payments import *
from cycle_finder import *

import networkx as nx

def get_success_ratio (Graph: nx.DiGraph, payment_dict: dict, debug: bool = False):
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
            make_payment(Graph, i, j, payment_dict[i,j])
            successful_payments += 1
            total_payments += 1
        except Exception as e:
            total_payments += 1
    return successful_payments/total_payments