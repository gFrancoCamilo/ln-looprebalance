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
        choose_payments(dataset, n_payments)
    except:
        raise Exception ('Failed to select payments')

#get_payment_dataset_test()
choose_payments_test(100)