import torch
from torch import nn
import torch.nn.functional as F

# Abstract classes for generator and discriminator

class Generator(nn.Module):
    def __init__(self, z_dim=100, out_ch=3, n_classes=1, emb_mode="concat", bn=True):
        super().__init__()
        self.z_dim = z_dim
        self.out_ch = out_ch
        self.n_classes = n_classes
        self.emb_mode = emb_mode
        self.bn = bn
        self.emb = nn.Embedding(self.n_classes, self.z_dim) if self.n_classes > 1 and self.emb_mode == "embed" else None

    def forward(self, z, y=None):
        raise NotImplementedError("Abstract method")

    def loss(self, d_output, device):
        raise NotImplementedError("Abstract method")

class Discriminator(nn.Module):
    def __init__(self, n_classes=0, emb_mode="concat", conditional_arch="CGAN", aux_loss_type="wasserstein", aux_loss_scalar=1):
        super().__init__()
        self.n_classes = n_classes
        self.emb_mode = emb_mode
        self.conditional_arch = conditional_arch
        self.aux_loss_scalar = aux_loss_scalar
        self.aux_loss_type = aux_loss_type

        if n_classes > 1:
            if emb_mode == "embed":
                self.emb = nn.Embedding(n_classes, size)

            if self.conditional_arch == "ACGAN":
                self.emb_mode = None

                if self.aux_loss_type == "cross_entropy":
                    self.aux_criterion = nn.CrossEntropyLoss()

    def forward(self, x, y=None, aux=True):
        raise NotImplementedError("Abstract method")

    def real_loss(self, output, device):
        raise NotImplementedError("Abstract method")

    def fake_loss(self, output, device):
        raise NotImplementedError("Abstract method")

    def aux_loss(self, output, labels, device):
        if self.aux_loss_type == "wasserstein":
            return self.aux_loss_scalar * torch.sum(torch.mul(F.one_hot(labels, self.n_classes)*(-2)+1, torch.sigmoid(output)) / torch.sum(F.one_hot(labels, self.n_classes), dim=0)[labels].unsqueeze(dim=1).expand_as(output))
        else:
            return self.aux_loss_scalar * self.aux_criterion(output, labels)
