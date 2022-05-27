from model.gru import *
from model.lstm import *
from model.rnn import *

def get_model(model, data):
    if model == 'RNN':
        if data == 'online_shopping_10_cats':
            return RNNClassifier
    elif model == 'GRU':
        if data == 'online_shopping_10_cats':
            return GRUClassifier
    raise ValueError('Model {} not supported'.format(model))