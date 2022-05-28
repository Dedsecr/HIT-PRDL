import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import random
import os
import torch

data_path = './data/jena_climate_2009_2016.csv'
data_dump_path = './data/jena_climate_2009_2016.pt'


def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0):
    n_features = ts.shape[1]
    X, Y = [], []

    for i in range(len(ts) - lag - n_ahead):
        Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
        X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], lag, n_features))
    # print('X shape: {}'.format(X.shape))
    # print('Y shape: {}'.format(Y.shape))
    return X, Y


def jena_climate_2009_2016(batch_size=64):
    print('Loading data...')
    if os.path.exists(data_dump_path):
        print('Loading data from {}'.format(data_dump_path))
        train_dataset, test_dataset = torch.load(data_dump_path)
    else:
        print('Loading data from {}'.format(data_path))
        data = pd.read_csv(data_path)

        data['Date Time'] = pd.to_datetime(data['Date Time'])
        data = data.resample('H', on='Date Time').mean()
        data = data.fillna(data.mean())
        for col in ['wv (m/s)', 'max. wv (m/s)']:
            data[col] = data[col].replace(-9999.00, 0)

        # Select features (columns) to be involved intro training and predictions
        cols = ['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)']
        data = data[cols]
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        data = data.values

        # Split data into train and test sets
        train_size = int(data.shape[0] * 0.75)
        train_data = data[:train_size, :]
        test_data = data[train_size:, :]

        # Split train data into X and Y
        train_X, train_y = create_X_Y(train_data,
                                      lag=5,
                                      n_ahead=2,
                                      target_index=0)
        test_X, test_y = create_X_Y(test_data,
                                    lag=5,
                                    n_ahead=2,
                                    target_index=0)

        # Create TensorDataset
        train_dataset = TensorDataset(torch.Tensor(train_X),
                                      torch.Tensor(train_y))
        test_dataset = TensorDataset(torch.Tensor(test_X),
                                     torch.Tensor(test_y))

        torch.save((train_dataset, test_dataset), data_dump_path)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    jena_climate_2009_2016()