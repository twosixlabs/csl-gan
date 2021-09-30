import torch
from torch import nn
import torch.nn.functional as F

from models import *
import util

class MNISTVanillaG(nn.Module):
    def __init__(self, latent_dim, n_classes=10):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

        self.latent_dim = latent_dim

        self.lin1 = nn.Linear(self.latent_dim + n_classes, 128)
        self.lin2 = nn.Linear(128, 784)

    def forward(self, z, y=None):
        x = z
        x = x if y is None else torch.cat([x, y], dim=1)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return torch.sigmoid(x).reshape(z.size(0), 1, 28, 28)

    def loss(self, d_output, device):
        return self.criterion(d_output, torch.ones(d_output.shape, device=device))

class MNISTVanillaD(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

        self.lin1 = nn.Linear(784 + n_classes, 128)
        self.lin2 = nn.Linear(128, 1)

    def forward(self, x, y=None):
        o = x.reshape(x.size(0), -1)
        o = o if y is None else torch.cat([o, y], dim=1)

        o = F.relu(self.lin1(o))
        o = self.lin2(o)
        return o

    def real_loss(self, output, device):
        return self.criterion(output, torch.ones(output.shape, device=device))

    def fake_loss(self, output, device):
        return self.criterion(output, torch.zeros(output.shape, device=device))

class MNIST_DCRN_G(DCResNetGenerator):
    def __init__(self, z_dim=128, channels=[128,128,64], first_filter_size=7, bn=True, n_classes=10):
        super().__init__(z_dim, channels, first_filter_size, bn, out_ch=1, n_classes=n_classes)

class MNIST_DCRN_D(DCResNetDiscriminator):
    def __init__(self, channels=[1, 64, 128], last_filter_size=7, n_classes=10):
        super().__init__(channels, last_filter_size, n_classes=n_classes)
