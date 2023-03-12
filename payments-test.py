from payments import *

def get_payment_dataset_test ():
    try:
        all_options = ['ripple','credit-card']
        for data_option in all_options:
            payments = get_payment_dataset(data_option)
            print(payments.describe())
    except:
        raise Exception ('Failed to load payments dataset')

def choose_payments_test (n_payments: int = 10):
    try:
        dataset = get_payment_dataset()
        payments_list = choose_payments(dataset, n_payments)
        return payments_list
    except:
        raise Exception ('Failed to select payments')

def get_payments_ln_test (Graph: nx.DiGraph, list_payments: list, n: int = 4):
    try:
        payments_dict = get_payments_ln (Graph, list_payments, n)
        for (i,j) in payments_dict:
            print ("Source: " + str(i))
            print ("Destination: " + str(j))
            print ("Value in satoshi: " + str(payments_dict[(i,j)]))
    except:
        raise Exception ("Could not select source and destination for payments")

#get_payment_dataset_test()
payment_list = choose_payments_test(100)
Graph = graph_names ('jul 2022')
Graph = validate_graph(Graph)

get_payments_ln_test(Graph, payment_list)