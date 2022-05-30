import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GAN_G(nn.Module):

    def __init__(self, n_noise=2, img_shape=(2, )):
        super(GAN_G, self).__init__()

        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*block(n_noise, 128, normalize=False),
                                   *block(128, 256), *block(256, 512),
                                   *block(512, 1024),
                                   nn.Linear(1024, int(np.prod(img_shape))),
                                   nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img


class GAN_D(nn.Module):

    def __init__(self, img_shape=(2, )):
        super(GAN_D, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
