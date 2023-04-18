import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def check_ripple_seasonality ():
    df = pd.read_csv('datasets/transactions-in-USD-jan-2013-aug-2016.txt')

    df = df.loc[df['USD_amount'] < 10**(8)].reset_index()
    df = df.loc[df['USD_amount'] > 0].reset_index()
    df = df.sort_values(by=['unix_timestamp'], ascending=True).reset_index(drop=True)

    initial_timestamp = 1388534400
    timestamp_iterator = 1388534400
    transactions_day = []
    day_accumulator = 0
    index = 0
    while timestamp_iterator < (initial_timestamp + 2*365*24*60*60):
        while df['unix_timestamp'][index] < initial_timestamp:
            index += 1
        while df['unix_timestamp'][index] < timestamp_iterator + (24*60*60):
            day_accumulator += df['USD_amount'][index]
            index += 1
        transactions_day.append(day_accumulator)
        timestamp_iterator = timestamp_iterator + (24*60*60)
        day_accumulator = 0
    
    transactions_week = []
    week_accumulator = 0
    index = 0
    for day in transactions_day:
        if index == 7:
            transactions_week.append(week_accumulator/index)
            week_accumulator = 0
            index = 0
        week_accumulator += day
        index += 1      

    plt.figure()
    plt.plot(transactions_day, lw=1.5)
    plt.ylabel('Valor das Transações (US$)', fontsize=16)
    plt.xlabel('Dia', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(16)
    plt.tight_layout()
    plt.savefig('results/transactions_day.png', dpi=600)

    plt.clf()
    plt.plot(transactions_week, lw=1.5)
    plt.ylabel('Valor das Transações (US$)', fontsize=16)
    plt.xlabel('Semana', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(16)
    plt.tight_layout()
    plt.savefig('results/transactions_week.png', dpi=600)
check_ripple_seasonality()