import torch
import torch.nn as nn
import torch.optim as optim
import data.caltech101 as caltech101
import model.alexnet as alexnet
from tqdm import tqdm
import argparse


def train(model, train_loader, optimizer, epochs, criterion, lr_scheduler, test_loader):
    model.train()
    acc_best = 0
    for epoch in range(epochs):

        for data, target in tqdm(train_loader):
            data, target = data.cuda(), target.cuda()
            data, target = data.view(-1, 784), target.view(-1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        print('Epoch {}/{} Loss: {:.4f} '.format(epoch, epochs, loss.item()))
        acc = test(model, test_loader)
        if acc > acc_best:
            acc_best = acc
            torch.save(model.state_dict(), './checkpoint/model.pt')
    print('Best accuracy: {:.2f}%'.format(acc_best*100))


def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = data.view(-1, 784), target.view(-1)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def main(batch_size, epochs, lr):
    train_loader, test_loader = caltech101.caltech101(batch_size)
    model = alexnet.AlexNext(10).cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().cuda()
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=epochs//5, gamma=0.1)

    train(model, train_loader, optimizer, epochs,
          criterion, lr_scheduler, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lab2')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    hidden_size = args.hidden_size
    main(batch_size, epochs, lr, hidden_size)
