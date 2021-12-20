import torch
from torch import nn
import torch.nn.functional as F

from models import *
from DCResNet_models import *
import util

class MNISTVanillaG(Generator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, out_ch=1)
        self.criterion = nn.BCEWithLogitsLoss()

        self.lin1 = nn.Linear(self.z_dim + self.n_classes, 128)
        self.lin2 = nn.Linear(128, 784*self.out_ch)

    def forward(self, z, y=None):
        x = z
        x = x if y is None else torch.cat([x, F.one_hot(y, num_classes=self.n_classes)], dim=1)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return torch.sigmoid(x).reshape(z.size(0), self.out_ch, 28, 28)

    def loss(self, d_output, device):
        return self.criterion(d_output, torch.ones(d_output.shape, device=device))

class MNISTVanillaD(Discriminator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss()

        if self.n_classes > 1 and self.aux_loss_type != "cross_entropy":
            raise Exception("Cross entropy loss is the only aux loss supported for vanilla architecture.")

        self.lin1 = nn.Linear(784 + self.n_classes, 128)
        self.lin2 = nn.Linear(128, 1)
        if self.n_classes > 1:
            self.linOutAux = nn.Linear(128, self.n_classes, bias=True) if self.conditional_arch == "ACGAN" else None

    def forward(self, x, y=None, aux=True):
        o = x.reshape(x.size(0), -1)
        o = o if y is None else torch.cat([o, F.one_hot(y, num_classes=self.n_classes)], dim=1)

        o = F.relu(self.lin1(o))
        return self.lin2(o), self.linOutAux(o) if aux and self.conditional_arch == "ACGAN" and self.n_classes > 1 else None

    def real_loss(self, output, device):
        return self.criterion(output, torch.ones(output.shape, device=device))

    def fake_loss(self, output, device):
        return self.criterion(output, torch.zeros(output.shape, device=device))

class MNIST_DCRN_G(DCResNetGenerator):
    def __init__(self, z_dim=128, channels=[128,128,64], first_filter_size=7, bn=True, n_classes=10, **kwargs):
        super().__init__(z_dim=z_dim, channels=channels, first_filter_size=first_filter_size, bn=bn, out_ch=1, n_classes=n_classes, **kwargs)

class MNIST_DCRN_D(DCResNetDiscriminator):
    def __init__(self, channels=[1, 64, 128], last_filter_size=7, n_classes=10, **kwargs):
        super().__init__(channels=channels, last_filter_size=last_filter_size, n_classes=n_classes, **kwargs)
