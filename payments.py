from pcn import *
from topology import *
import pandas as pd
import numpy as np
from scipy import stats

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
    payments = payments[(np.abs(stats.zscore(payments)) < 3)]
    return payments
