from model.gru import *
from model.lstm import *
from model.rnn import *
from model.bilstm import *


def get_model(model):
    if model == 'RNN':
        return RNNRegressor
    elif model == 'GRU':
        return GRURegressor
    elif model == 'LSTM':
        return LSTMRegressor
    elif model == 'BiLSTM':
        return BiLSTMRegressor
    raise ValueError('Model {} not supported'.format(model))