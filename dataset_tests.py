import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def check_ripple_seasonality ():
    df = pd.read_csv('datasets/transactions-in-USD-jan-2013-aug-2016.txt')

    df = df.loc[df['USD_amount'] < 10**(8)].reset_index()
    df = df.loc[df['USD_amount'] > 0].reset_index()
    df = df.sort_values(by=['unix_timestamp'], ascending=True).reset_index(drop=True)

    initial_timestamp = df['unix_timestamp'][0]
    timestamp_iterator = df['unix_timestamp'][0]
    transactions_day = []
    day_accumulator = 0
    index = 0
    while timestamp_iterator < (initial_timestamp + 365*24*60*60):
        while df['unix_timestamp'][index] < timestamp_iterator + (24*60*60):
            day_accumulator += df['unix_timestamp'][index]
            index += 1
        transactions_day.append(day_accumulator)
        timestamp_iterator = timestamp_iterator + (24*60*60)
        day_accumulator = 0
    
    plt.plot(transactions_day)
    plt.show()
check_ripple_seasonality()