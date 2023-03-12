from results import *

def get_success_ratio_test (Graph: nx.DiGraph, payment_dict: list):
    try:
        result = get_success_ratio(Graph, payment_dict)
        print("Payment Success Ratio: " + str(result))
    except:
        raise Exception ('Failed to compute payment success ratio')

Graph = graph_names ('jul 2022')
Graph = validate_graph(Graph)

Graph = set_balance_ln(Graph, 0.01)
payment_dataset = get_payment_dataset()
payment_list = choose_payments(payment_dataset, 100)
payment_dict = get_payments_ln(Graph, payment_list)
get_success_ratio_test(Graph, payment_dict)