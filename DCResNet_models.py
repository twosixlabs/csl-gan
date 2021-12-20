import torch
from torch import nn
import torch.nn.functional as F

import util
from models import *

class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch, filter_size, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, filter_size, padding="same", bias=bias)

    def forward(self, x):
        o = torch.cat([x, x, x, x], 1)
        o = F.pixel_shuffle(o, 2)
        o = self.conv(o)
        return o

class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch, filter_size, bn=True):
        super().__init__()
        self.shortcut = UpsampleConv(in_ch, out_ch, 1)
        self.bn1 = nn.BatchNorm2d(in_ch) if bn else nn.GroupNorm(32, in_ch)
        self.convUp = UpsampleConv(in_ch, out_ch, filter_size, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch) if bn else nn.GroupNorm(32, out_ch)
        self.conv = nn.Conv2d(out_ch, out_ch, filter_size, padding="same")

    def forward(self, x):
        s = self.shortcut(x)

        o = self.bn1(x)
        o = F.relu(o)
        o = self.convUp(o)
        o = self.bn2(o)
        o = F.relu(o)
        o = self.conv(o)

        return o + s

class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch, filter_size, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, filter_size, padding="same", bias=bias)

    def forward(self, x):
        o = torch.cat([x, x, x, x], 1)
        o = F.pixel_shuffle(o, 2)
        o = self.conv(o)
        return o

class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch, filter_size, bn=True):
        super().__init__()
        self.shortcut = UpsampleConv(in_ch, out_ch, 1)
        self.bn1 = nn.BatchNorm2d(in_ch) if bn else nn.GroupNorm(32, in_ch)
        self.convUp = UpsampleConv(in_ch, out_ch, filter_size, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch) if bn else nn.GroupNorm(32, out_ch)
        self.conv = nn.Conv2d(out_ch, out_ch, filter_size, padding="same")

    def forward(self, x):
        s = self.shortcut(x)

        o = self.bn1(x)
        o = F.relu(o)
        o = self.convUp(o)
        o = self.bn2(o)
        o = F.relu(o)
        o = self.conv(o)

        return o + s

class DCResNetGenerator(Generator):
    def __init__(self, channels, first_filter_size, **kwargs):
        super().__init__(**kwargs)
        self.first_filter_size = first_filter_size

        self.linIn = nn.Linear(self.z_dim + (self.n_classes if self.emb_mode == "concat" else 0), self.first_filter_size**2*channels[0])

        self.blocks = []
        for i in range(1, len(channels)):
            self.blocks.append(ResBlockUp(channels[i-1], channels[i], 5, bn=self.bn))
        self.blocks = nn.ModuleList(self.blocks)

        self.bn = nn.BatchNorm2d(channels[-1]) if self.bn else nn.GroupNorm(32, channels[-1])
        self.convOut = nn.Conv2d(channels[-1], self.out_ch, 3, padding="same")

    def forward(self, z, y=None):
        x = z
        if not y is None:
            if self.emb_mode == "embed":
                x = torch.mul(z, self.emb(y))
            elif self.emb_mode == "concat":
                x = torch.cat((z, F.one_hot(y, self.n_classes)), dim=1)

        x = self.linIn(x)
        x = x.reshape(z.size(0), -1, self.first_filter_size, self.first_filter_size)

        for block in self.blocks:
            x = block(x)

        x = self.bn(x)
        x = F.relu(x)
        x = self.convOut(x)
        return torch.tanh(x)

    def loss(self, d_output, device):
        return -torch.mean(d_output)

class DCResNetDiscriminator(Discriminator):
    def __init__(self, channels, last_filter_size, **kwargs):
        super().__init__(**kwargs)
        self.blocks = []

        if self.emb_mode == "concat" and self.n_classes > 1:
            channels[0] += self.n_classes

        for i in range(1, len(channels)):
            self.blocks.append(nn.Conv2d(channels[i-1], channels[i], 5, stride=2, padding=2))
        self.blocks = nn.ModuleList(self.blocks)

        size = channels[-1]*last_filter_size**2

        if self.n_classes < 2 or self.conditional_arch != "WCGAN":
            self.linOut = nn.Linear(size, 1, bias=False)
        if self.n_classes > 1 and self.conditional_arch in ["ACGAN", "WCGAN"]:
            self.linOutAux = nn.Linear(size, self.n_classes, bias=True)

    def forward(self, x, y=None, aux=True):
        # Concat like https://cameronfabbri.github.io/papers/conditionalWGAN.pdf
        o = torch.cat((x, F.one_hot(y, self.n_classes).view(x.size(0), -1, 1, 1).expand(-1, -1, x.size(2), x.size(3))), dim=1) if self.emb_mode == "concat" and self.n_classes > 1 else x
        for block in self.blocks:
            o = F.leaky_relu(block(o), 0.2)

        o = o.reshape(x.size(0), -1)

        if not y is None:
            if self.emb_mode == "embed":
                o = torch.mul(self.emb(y), o)

        out_aux = self.linOutAux(o) if aux and hasattr(self, "linOutAux") else None
        if not out_aux is None and self.conditional_arch == "WCGAN":
            out = (out_aux * F.one_hot(y, self.n_classes)).sum(dim=1)
            #print("Main:",F.one_hot(y, self.n_classes).float().mean(dim=0)/2)
        else:
            out = self.linOut(o)

        return out, out_aux

    def real_loss(self, output, device):
        return -torch.mean(output)

    def fake_loss(self, output, device):
        return torch.mean(output)
