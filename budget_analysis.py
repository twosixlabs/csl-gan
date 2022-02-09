import numpy as np
import argparse
from pathlib import Path
import torch
from torch import nn
import torch.optim as optim

import opacus
from opacus import PrivacyEngine, ISPrivacyEngine, TMPrivacyEngine, SVPrivacyEngine

import options
import util
import init_util


parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to output folder containing opt.txt")
parser.add_argument("epochs", type=int)
opt_new = parser.parse_args()

opt_new.path = util.add_slash(opt_new.path)
opt = options.load_opt(opt_new.path + "opt.txt")

class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.something = nn.Linear(1, 1)

    def forward(self, x):
        return self.something(x)

D = Disc()
d_optimizer = optim.Adam(D.parameters(), lr=opt.d_lr, betas=(opt.adam_b1, opt.adam_b2))

privacy_engine = None
privacy_params = {
    "batch_size": opt.batch_size,
    "sample_size": opt.train_set_size,
    "alphas": [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 1200)), # Recommended by Opacus
    "noise_multiplier": opt.sigma,
}
if opt.dp_mode == "is":
    privacy_engine = ISPrivacyEngine(
        D, **privacy_params,
        per_param=opt.imm_sens_per_param,
        scaling_vec=None if opt.imm_sens_scaling_mode == "standard" else opt.imm_sens_scaling_vec
    )
elif opt.dp_mode == "gc":
    opt.clipping_param_per_layer = [1 for _ in range(len(list(D.parameters())))] if opt.clipping_param_per_layer is None else opt.clipping_param_per_layer
    privacy_engine = PrivacyEngine(
        D, **privacy_params,
        accum_passes=not opt.grad_clip_split,
        num_private_passes=1 if opt.grad_clip_split else None,
        auto_clip_and_accum_on_step=False,
        max_grad_norm=opt.clipping_param_per_layer if opt.grad_clip_mode[-3:] == "-pl" else opt.clipping_param
    )
    privacy_engine.disable_hooks()
elif opt.dp_mode == "tm":
    privacy_engine = TMPrivacyEngine(
        D, **privacy_params,
        smooth_sens_= opt.smooth_sens_t,
        m_trim = opt.tm_m,
        min_val = opt.tm_max_val,
        max_val = opt.tm_min_val,
        sens_compute_bs = opt.batch_size * 2 if opt.tm_sens_compute_bs is None else opt.tm_sens_compute_bs,
        rho_per_epoch = opt.tm_rho_per_epoch
    )
elif opt.dp_mode == "sv":
    privacy_engine = SVPrivacyEngine(
        D, **privacy_params,
        smooth_sens_= opt.smooth_sens_t,
        rho_per_epoch = opt.tm_rho_per_epoch
    )

privacy_engine.attach(d_optimizer)
privacy_engine._set_seed(opt.manual_seed)


privacy_engine.steps = (60000 if opt.dataset == "MNIST" else 202599) * opt_new.epochs / opt.batch_size
print(privacy_engine.get_privacy_spent())
