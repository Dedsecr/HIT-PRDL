import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import transforms
import os
import torch
import cv2
from glob import glob
import math
from tqdm import tqdm
import re

data_path = './data/plant-seedlings-classification/'


def preprocess(images):
    images_processed = []
    for img in images:
        # Use gaussian blur
        blurImg = cv2.GaussianBlur(img, (5, 5), 0)

        # Convert to HSV image
        hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)

        # Create mask (parameters - green color range)
        # lower_green = (25, 40, 50)
        lower_green = (35, 43, 46)
        upper_green = (77, 255, 255)
        mask = cv2.inRange(hsvImg, lower_green, upper_green)

        # Create bool mask
        bMask = mask > 0

        # Apply the mask
        clear = np.zeros_like(img, np.uint8)  # Create empty image
        clear[bMask] = img[bMask]  # Apply boolean mask to the origin image

        images_processed.append(clear)  # Append image without backgroung

    images_processed = np.array(images_processed)
    return images_processed


'''
random rotate [-180, 180],
random shift 0.3,
random scale 0.3,
random horizontal and vertical flip
'''


def load_dataset(files, img_size, label, pre):
    trainImg = []
    trainLabel = []
    testId = []

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
    ])

    for img in tqdm(files):
        trainImg.append(
            cv2.resize(cv2.imread(img),
                       (img_size, img_size)))  # Get image (with resizing)
        if label:
            trainLabel.append(re.split(
                '/|\\\\', img)[-2])  # Get image label (folder name)
        else:
            testId.append(re.split('/|\\\\', img)[-1])  # Images id's

    trainImg = np.array(trainImg)  # Train images set

    # Preprocess images
    if pre:
        trainImg = preprocess(trainImg)
    trainImg = torch.Tensor(trainImg).permute(0, 3, 1, 2)

    if label:
        trainLabel = np.array(trainLabel)  # Train labels set
        specie2idx = {
            specie: idx
            for idx, specie in enumerate(np.unique(trainLabel))
        }
        idx2specie = {
            idx: specie
            for idx, specie in enumerate(np.unique(trainLabel))
        }

        trainLabel = np.array([specie2idx[specie] for specie in trainLabel])
        trainLabel = torch.from_numpy(trainLabel).long()
        dataset = TensorDataset(transform(trainImg), trainLabel)
        return dataset, idx2specie
    else:
        dataset = TensorDataset(trainImg)
        return dataset, testId


def plant_seedlings(img_size=224, batch_size=256, pre=False):
    print('Loading data...')
    data_dump_path = './data/plant-seedlings-classification_{}_{}.pt'.format(
        img_size, pre)
    if os.path.exists(data_dump_path):
        print('Loading data from {}'.format(data_dump_path))
        train_data, val_data, test_data, testId, idx2specie = torch.load(
            data_dump_path)
    else:
        print('Loading data from {}'.format(data_path))
        path_train = data_path + 'train/*/*.png'
        path_test = data_path + 'test/*.png'
        files_train = glob(path_train)
        files_test = glob(path_test)

        dataset, idx2specie = load_dataset(files_train, img_size, label=True, pre=pre)
        train_data, val_data = random_split(
            dataset, [len(dataset) - len(dataset) // 10,
                      len(dataset) // 10])
        test_data, testId = load_dataset(files_test, img_size, label=False, pre=pre)

        torch.save((train_data, val_data, test_data, testId, idx2specie),
                   data_dump_path)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, testId, idx2specie


if __name__ == '__main__':
    train_loader, val_loader, test_loader, testId, idx2specie = plant_seedlings(
    )
    print(idx2specie)