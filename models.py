import torch
from torch import nn
import torch.nn.functional as F

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

class DCResNetGenerator(nn.Module):
    def __init__(self, z_dim, channels, first_filter_size, bn, out_ch=3, n_classes=0):
        super().__init__()
        self.first_filter_size = first_filter_size

        self.linIn = nn.Linear(z_dim + n_classes, self.first_filter_size**2*channels[0])

        self.blocks = []
        for i in range(1, len(channels)):
            self.blocks.append(ResBlockUp(channels[i-1], channels[i], 5, bn=bn))
        self.blocks = nn.ModuleList(self.blocks)

        self.bn = nn.BatchNorm2d(channels[-1]) if bn else nn.GroupNorm(32, channels[-1])
        self.convOut = nn.Conv2d(channels[-1], out_ch, out_ch, padding="same")


    def forward(self, z, y=None):
        x = z if y is None else torch.cat((z, y), dim=1)
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

class DCResNetDiscriminator(nn.Module):
    def __init__(self, channels, last_filter_size, n_classes=0):
        super().__init__()
        self.blocks = []

        for i in range(1, len(channels)):
            self.blocks.append(nn.Conv2d(channels[i-1], channels[i], 5, stride=2, padding=2))
        self.blocks = nn.ModuleList(self.blocks)

        size = channels[-1]*last_filter_size**2 + n_classes

        if n_classes > 0:
            self.linOut = nn.Sequential(
                nn.Linear(size, size//5),
                nn.LeakyReLU(0.2),
                nn.Linear(size//5, 1)
            )
        else:
            self.linOut = nn.Linear(size, 1, bias=False)

    def forward(self, x, y=None):
        o = x
        for block in self.blocks:
            o = F.leaky_relu(block(o), 0.2)

        o = o.reshape(x.size(0), -1)
        o = o if y is None else torch.cat((o, y), dim=1)
        o = self.linOut(o)
        return o

    def real_loss(self, output, device):
        return -torch.mean(output)

    def fake_loss(self, output, device):
        return torch.mean(output)
