# https://github.com/alps-lab/dpgan/tree/master/models/gans

import torch
from torch import nn
import torch.nn.functional as F

from models import *
import util

class CelebA_DCRN_G64(DCResNetGenerator):
    def __init__(self, z_dim=128, channels=[512,512,256,128,64], first_filter_size=4, bn=True, n_classes=0):
        super().__init__(z_dim, channels, first_filter_size, bn, out_ch=3, n_classes=n_classes)

class CelebA_DCRN_D64(DCResNetDiscriminator):
    def __init__(self, channels=[3, 64, 128, 256, 512], last_filter_size=4, n_classes=0):
        super().__init__(channels, last_filter_size, n_classes=n_classes)

class CelebA_DCRN_G48(DCResNetGenerator):
    def __init__(self, z_dim=128, channels=[512,512,256,128], first_filter_size=6, bn=True, n_classes=0):
        super().__init__(z_dim, channels, first_filter_size, bn, out_ch=3, n_classes=n_classes)

class CelebA_DCRN_D48(DCResNetDiscriminator):
    def __init__(self, channels=[3,128,256,512], last_filter_size=6, n_classes=0):
        super().__init__(channels, last_filter_size, n_classes=n_classes)
