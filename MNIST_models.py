import torch
from torch import nn
import torch.nn.functional as F

class MNISTVanillaG(nn.Module):
    def __init__(self, latent_dim, unconditional=False, num_classes=10):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

        self.latent_dim = latent_dim
        self.unconditional = unconditional

        self.lin1 = nn.Linear(self.latent_dim + (0 if unconditional else num_classes), 128)
        self.lin2 = nn.Linear(128, 784)

    def forward(self, z, y=None):
        x = z
        x = x if self.unconditional else torch.cat([x, y], dim=1)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return torch.sigmoid(x).reshape(z.size(0), 1, 28, 28)

    def loss(self, d_output, device):
        return self.criterion(d_output, torch.ones(d_output.shape, device=device))

class MNISTVanillaD(nn.Module):
    def __init__(self, unconditional=False, num_classes=10):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

        self.unconditional = unconditional

        self.lin1 = nn.Linear(784 + (0 if unconditional else num_classes), 128)
        self.lin2 = nn.Linear(128, 1)

    def forward(self, x, y=None):
        o = x.reshape(x.size(0), -1)
        o = o if self.unconditional else torch.cat([o, y], dim=1)

        o = F.relu(self.lin1(o))
        o = self.lin2(o)
        return o

    def real_loss(self, output, device):
        return self.criterion(output, torch.ones(output.shape, device=device))

    def fake_loss(self, output, device):
        return self.criterion(output, torch.zeros(output.shape, device=device))
