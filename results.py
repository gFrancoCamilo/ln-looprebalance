from topology import *
from pcn import *
from payments import *
from cycle_finder import *

import networkx as nx

def get_success_ratio (Graph: nx.DiGraph, payment_dict: dict, debug: bool = False):
    total_payments = 0
    successful_payments = 0
    for (i,j) in payment_dict:
        try:
            make_payment(Graph, i, j, payment_dict[i,j])
            successful_payments += 1
            total_payments += 1
        except Exception as e:
            total_payments += 1
    return successful_payments/total_payments