import torch
import torchvision


def caltech101(batch_size):
    """
    Caltech 101 dataset
    """
    train_dataset = torchvision.datasets.Caltech101(
        root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.Caltech101(
        root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader