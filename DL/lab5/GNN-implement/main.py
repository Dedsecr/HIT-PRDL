import torch
import torch.nn as nn
import torch.optim as optim
from data import *
from model import *
from tqdm import tqdm
import argparse
from train import *
import random


def main(model, batch_size, epochs, lr, n_noise, clamp):
    data_loader = points(batch_size)
    if model == 'GAN':
        model_G = GAN_G()
        model_D = GAN_D()
        model_G, model_D = model_G.cuda(), model_D.cuda()
        optimizer_D = optim.Adam(model_D.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_G = optim.Adam(model_G.parameters(), lr=lr, betas=(0.5, 0.999))
        train_GAN(model_G, model_D, data_loader, optimizer_G, optimizer_D, epochs, n_noise)
    elif model == 'WGAN':
        model_G = WGAN_G()
        model_D = WGAN_D()
        model_G, model_D = model_G.cuda(), model_D.cuda()
        optimizer_D = optim.RMSprop(model_D.parameters(), lr=lr)
        optimizer_G = optim.RMSprop(model_G.parameters(), lr=lr)
        train_WGAN(model_G, model_D, data_loader, optimizer_G, optimizer_D, epochs, n_noise, clamp)
    elif model == 'WGAN-GP':
        model_G = WGAN_G()
        model_D = WGAN_D()
        model_G, model_D = model_G.cuda(), model_D.cuda()
        optimizer_D = optim.Adam(model_D.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_G = optim.Adam(model_G.parameters(), lr=lr, betas=(0.5, 0.999))
        train_WGAN_GP(model_G, model_D, data_loader, optimizer_G, optimizer_D, epochs, n_noise, clamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lab5')
    parser.add_argument('--model',
                        type=str,
                        default='WGAN',
                        choices=['GAN', 'WGAN', 'WGAN-GP'],
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
                        default=1e-5,
                        help='learning rate (default: 1e-5)')
    args = parser.parse_args()
    n_noise = 2
    clamp = 0.01
    main(args.model, args.batch_size, args.epochs, args.lr,
         n_noise, clamp)
