import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
import random

path = './data/'


def jena_climate_2009_2016(batch_size=1):
    d = pd.read_csv('data/jena_climate_2009_2016.csv')

    print(f'Shape of data: {d.shape}')


    # Converting the dt column to datetime object
    d['dt'] = [datetime.datetime.utcfromtimestamp(x) for x in d['dt']]

    # Sorting by the date
    d.sort_values('dt', inplace=True)
    # Listing the min and the max dates 
    print(f"First date {min(d['dt'])}")
    print(f"Most recent date {max(d['dt'])}")


if __name__ == '__main__':
    jena_climate_2009_2016()