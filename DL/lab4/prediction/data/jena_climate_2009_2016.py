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


def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0, train=False):
    n_features = ts.shape[1]
    X, Y = [], []
    if train:
        for i in range(0, len(ts) - lag - n_ahead):
            X.append(ts[i:(i + lag)])
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
    else:
        for i in range(0, len(ts) - lag - n_ahead, lag + n_ahead):
            X.append(ts[i:(i + lag)])
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])

    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], lag, n_features))
    # print('X shape: {}'.format(X.shape))
    # print('Y shape: {}'.format(Y.shape))
    return X, Y


def jena_climate_2009_2016(batch_size=64):
    print('Loading data...')
    if os.path.exists(data_dump_path):
        print('Loading data from {}'.format(data_dump_path))
        train_dataset, test_dataset, Temp_mean, Temp_std = torch.load(
            data_dump_path)
    else:
        print('Loading data from {}'.format(data_path))
        data = pd.read_csv(data_path)

        # data['Date Time'] = [
        #     datetime.datetime.strptime(x, "%d.%m.%Y %H:%M:%S")
        #     for x in data['Date Time']
        # ]
        # # data = data.resample('H', on='Date Time').mean()
        # data = data.fillna(data.mean())
        # for col in ['wv (m/s)', 'max. wv (m/s)']:
        #     data[col] = data[col].replace(-9999.00, 0)

        # Select features (columns) to be involved intro training and predictions
        # cols = ['T (degC)', 'p (mbar)', 'rh (%)', 'wv (m/s)']
        # cols = ['T (degC)', 'p (mbar)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)']
        cols = ['T (degC)']
        data = data[cols]

        # Temp_mean, Temp_std = data.mean(axis=0)['T (degC)'], data.std(
        #     axis=0)['T (degC)']
        Temp_mean, Temp_std = 0, 1

        # data = (data - data.mean(axis=0)) / data.std(axis=0)
        data = data.values

        # Split data into train and test sets
        train_size = int(data.shape[0] * 0.75)
        train_data = data[:train_size, :]
        test_data = data[train_size:, :]

        # Split train data into X and Y
        train_X, train_y = create_X_Y(train_data,
                                      lag=5 * 24 * 6,
                                      n_ahead=1,
                                      target_index=0,
                                      train=True)
        test_X, test_y = create_X_Y(test_data,
                                    lag=5 * 24 * 6,
                                    n_ahead=2 * 24 * 6,
                                    target_index=0)

        # Create TensorDataset
        train_dataset = TensorDataset(torch.Tensor(train_X),
                                      torch.Tensor(train_y))
        test_dataset = TensorDataset(torch.Tensor(test_X),
                                     torch.Tensor(test_y))

        torch.save((train_dataset, test_dataset, Temp_mean, Temp_std),
                   data_dump_path)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, Temp_mean, Temp_std


if __name__ == '__main__':
    # train_loader, test_loader = jena_climate_2009_2016()
    std = 8.409366
    mean = 9.442390
    path = 'data/preds_targets_RNN.pt'
    preds, targets = torch.load(path)
    train_dataset, test_dataset = torch.load(data_dump_path)
    data_id = 45
    print(preds.shape, targets.shape)
    print(test_dataset.tensors[0].shape, test_dataset.tensors[1].shape)
    print(preds[data_id] * std + mean, targets[data_id] * std + mean)
    print(test_dataset.tensors[0][data_id].numpy() * std + mean,
          test_dataset.tensors[1][data_id].numpy() * std + mean)
