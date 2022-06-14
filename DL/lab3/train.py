import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from utils import *


def train(model, train_loader, optimizer, epochs, criterion, lr_scheduler,
          val_loader, res_path, cuda):
    # Set model to training mode
    model.train()
    # Initialize accuracy
    train_acc_best = 0
    val_acc_best = 0
    # Iterate over epochs
    for epoch in range(epochs):
        model.train()
        # Initialize loss
        Loss = AverageMeter()
        Acc = AverageMeter()
        # Iterate over batches
        for data, target in tqdm(train_loader):
            # Move data to GPU
            if cuda:
                data, target = data.cuda(), target.cuda()
            # Zero out gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate accuracy
            acc = accuracy(output.data, target, topk=(1, ))[0]
            # Calculate loss
            loss = criterion(output, target)
            Loss.update(loss.item(), data.size(0))
            Acc.update(acc.item(), data.size(0))
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
        # Update learning rate
        lr_scheduler.step()
        # Print loss and accuracy
        print('Epoch: {}/{}, Loss: {:.2f}, Accuracy: {:.2f}'.format(
            epoch, epochs, Loss.avg, Acc.avg))
        if Acc.avg > train_acc_best:
            train_acc_best = Acc.avg
        # Evaluate model
        val_acc = validate(model, val_loader, criterion, cuda)
        # Save model if accuracy is best
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            torch.save(model.state_dict(), res_path + 'model.pth')
    print('Best accuracy: {:.2f}'.format(val_acc_best))
    with open(res_path + 'acc.txt', 'w') as f:
        f.write('Best train acc: {:.2f}\n'.format(train_acc_best))
        f.write('Best val acc: {:.2f}\n'.format(val_acc_best))


def validate(model, val_loader, criterion, cuda):
    # Set model to evaluation mode
    model.eval()
    # Initialize accuracy
    Acc = AverageMeter()
    Loss = AverageMeter()
    with torch.no_grad():
        for data, target in val_loader:
            # Move data to GPU
            if cuda:
                data = data.cuda()
                target = target.cuda()
            # Forward pass
            output = model(data)
            # Calculate accuracy
            acc = accuracy(output, target, topk=(1, ))[0]
            # Calculate loss
            loss = criterion(output, target)
            Loss.update(loss.item(), data.size(0))
            Acc.update(acc.item(), data.size(0))

    print('Test set: Accuracy: {:.2f}'.format(Acc.avg))
    return Acc.avg


def predict(model, test_loader, cuda):
    # Set model to evaluation mode
    model.eval()
    results = []
    with torch.no_grad():
        for data in test_loader:
            data = data[0]
            # Move data to GPU
            if cuda:
                data = data.cuda()
            # Forward pass
            output = model(data)
            # Get predictions
            pred = output.data.max(1, keepdim=True)[1]
            results.append(pred.cpu().numpy())
    return np.concatenate(results)