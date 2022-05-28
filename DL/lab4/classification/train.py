import torch
import torch.nn as nn
import torch.optim as optim
from data import *
from model import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np


def get_metrics(preds, targets):
    acc = accuracy_score(targets, preds)
    recall = recall_score(targets, preds, average='macro')
    f1 = f1_score(targets, preds, average='macro')
    return acc, recall, f1


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
    print('\tBest accuracy: {:.4f}'.format(metrics_best[0]), end=' ')
    print('Best recall: {:.4f}'.format(metrics_best[1]), end=' ')
    print('Best f1: {:.4f}'.format(metrics_best[2]))


def test(model, test_loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    metrics = get_metrics(preds, targets)
    print('Test result:')
    print('\tAccuracy: {:.4f}'.format(metrics[0]), end=' ')
    print('Recall: {:.4f}'.format(metrics[1]), end=' ')
    print('F1: {:.4f}'.format(metrics[2]))
    return metrics