import torch
import torch.nn as nn
import torch.optim as optim
from data import *
from model import *
from tqdm import tqdm
import argparse


def train(model, train_loader, optimizer, epochs, criterion, lr_scheduler,
          test_loader):
    model.train()
    acc_best = 0
    for epoch in range(epochs):

        for data, target in tqdm(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            # pred = output.data.max(1, keepdim=True)[1]
            # print(data)
            optimizer.step()

        lr_scheduler.step()
        print('Epoch {}/{} Loss: {:.4f} '.format(epoch, epochs, loss.item()))
        acc = test(model, test_loader)
        if acc > acc_best:
            acc_best = acc
            torch.save(model.state_dict(), './checkpoint/model.pt')
    print('Best accuracy: {:.2f}%'.format(acc_best * 100))


def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output, _ = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)