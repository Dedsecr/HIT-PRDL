import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from data.process_word import process_word
import os
import matplotlib.pyplot as plt

path = './data/online_shopping_10_cats/'
data_path = path + 'online_shopping_10_cats.csv'


def get_tensor_data(data):
    data_X = torch.Tensor([x for x in data['review_vec']])
    data_y = torch.Tensor(data['cat_id'].values).long()
    return data_X, data_y


def online_shopping_10_cats(batch_size=64, max_length=32, embedding_dim=32):
    print('Loading data...')
    data_dump_path = path + 'online_shopping_10_cats_{}_{}.pt'.format(
        max_length, embedding_dim)
    if os.path.exists(data_dump_path):
        print('Loading data from {}'.format(data_dump_path))
        train_dataset, val_dataset, test_dataset = torch.load(data_dump_path)
    else:
        print('Loading data from {}'.format(data_path))
        data = pd.read_csv(data_path)

        data['cat_id'] = data['cat'].factorize()[0]
        # id_to_cat = dict(data[['cat_id', 'cat']].values)
        data = process_word(data, 'review', path, max_length, embedding_dim)

        data_groups = list(data.groupby(lambda x: x % 5))

        data_train = pd.concat(
            [data_groups[1][1], data_groups[2][1], data_groups[3][1]])
        data_val = data_groups[4][1]
        data_test = data_groups[0][1]

        train_X, train_y = get_tensor_data(data_train)
        val_X, val_y = get_tensor_data(data_val)
        test_X, test_y = get_tensor_data(data_test)

        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        test_dataset = TensorDataset(test_X, test_y)

        torch.save((train_dataset, val_dataset, test_dataset),
                   data_dump_path)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)
    print('Data loaded.')
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = online_shopping_10_cats()
    for data in train_loader:
        print(data)
        break