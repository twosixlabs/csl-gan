# https://github.com/alps-lab/dpgan/tree/master/models/gans

import torch
from torch import nn
import torch.nn.functional as F

from DCResNet_models import *
import util

class CelebA_DCRN_G64(DCResNetGenerator):
    def __init__(self, z_dim=128, channels=[512,512,256,128,64], first_filter_size=4, **kwargs):
        super().__init__(z_dim=z_dim, channels=channels, first_filter_size=first_filter_size, out_ch=3, **kwargs)

class CelebA_DCRN_D64(DCResNetDiscriminator):
    def __init__(self, channels=[3, 64, 128, 256, 512], last_filter_size=4, **kwargs):
        super().__init__(channels=channels, last_filter_size=last_filter_size, **kwargs)

class CelebA_DCRN_G48(DCResNetGenerator):
    def __init__(self, z_dim=128, channels=[512,512,256,128], first_filter_size=6, **kwargs):
        super().__init__(z_dim=z_dim, channels=channels, first_filter_size=first_filter_size, out_ch=3, **kwargs)

class CelebA_DCRN_D48(DCResNetDiscriminator):
    def __init__(self, channels=[3,128,256,512], last_filter_size=6, **kwargs):
        super().__init__(channels=channels, last_filter_size=last_filter_size, **kwargs)
