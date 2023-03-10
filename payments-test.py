from payments import *

def get_payment_dataset_test ():
    try:
        all_options = ['ripple','credit-card']
        for data_option in all_options:
            payments = get_payment_dataset(data_option)
            print(payments.describe())
    except:
        raise Exception ('Failed to load payments dataset')

get_payment_dataset_test()