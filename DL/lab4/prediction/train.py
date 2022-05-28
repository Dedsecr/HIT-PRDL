import torch
import torch.nn as nn
import torch.optim as optim
from data import *
from model import *
from tqdm import tqdm
from sklearn.metrics import median_absolute_error, mean_absolute_error
import numpy as np


def get_metrics(preds, targets):
    # preds = preds.flatten()
    # targets = targets.flatten()
    # print(preds.shape, targets.shape)
    # print('preds:', preds)
    # print('targets:', targets)
    mae = mean_absolute_error(preds, targets)
    mre = median_absolute_error(preds, targets)
    return mae, mre


def train(model, train_loader, optimizer, epochs, criterion, lr_scheduler,
          test_loader):
    metrics_best = (0, 0, 0)
    for epoch in range(epochs):
        model.train()
        for data, target in tqdm(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        print('Epoch {}/{} Loss: {:.4f} '.format(epoch, epochs, loss.item()))
        metrics = test(model, test_loader)
        if metrics[0] > metrics_best[0]:
            metrics_best = metrics
            torch.save(model.state_dict(), './checkpoint/model.pt')
    print('Best result:')
    print('\tMAE: {:.4f}'.format(metrics_best[0]), end=' ')
    print('MRE: {:.4f}'.format(metrics_best[1]))


def test(model, test_loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            pred = model(data)
            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    metrics = get_metrics(preds, targets)
    print('Test result:')
    print('\tMAE: {:.4f}'.format(metrics[0]), end=' ')
    print('MRE: {:.4f}'.format(metrics[1]))
    return metrics