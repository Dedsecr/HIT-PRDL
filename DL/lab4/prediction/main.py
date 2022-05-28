import torch
import torch.nn as nn
import torch.optim as optim
from data import *
from model import *
from tqdm import tqdm
import argparse
from train import *


def main(model, batch_size, epochs, lr):
    train_loader, test_loader = jena_climate_2009_2016(
        batch_size)
    model = model()
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss().cuda()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=epochs // 5,
                                             gamma=0.1)
    train(model, train_loader, optimizer, epochs, criterion, lr_scheduler,
          test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lab4')
    parser.add_argument('--model',
                        type=str,
                        default='RNN',
                        choices=['RNN', 'GRU', 'LSTM', 'BiLSTM'],
                        help='model name')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate (default: 0.001)')
    args = parser.parse_args()
    model = get_model(args.model)
    main(model, args.batch_size, args.epochs, args.lr)
