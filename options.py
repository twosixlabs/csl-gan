import argparse
from argparse import Namespace
from datetime import datetime
import json
import os
import random
import torch

import util

MNIST_DEFAULTS = {
    "data_path": "/persist/datasets/mnist/",
    "model": "Vanilla",
    "im_size": 28,
    "n_epochs": 10000,
    "g_lr": 0.0002,
    "d_lr": 0.0002,
    "batch_size": 600,
    "batch_split_size": 60,
    "train_set_size": 60000,
    "g_latent_dim": 100,
    "n_d_steps": 1,
    "adam_b1": 0.9,
    "adam_b2": 0.999,
    "penalty": [],
    "delta": 1e-5,
    "sigma": 5.0,
    "grad_clip_mode": "standard",
    "C": 4.0,
    "C_weight": 10.0,
    "C_bias": 1.0,
    "save_every": 50,
    "log_every": 100000, # Gets rounded down to be 1 epoch
    "sample_every": 1200000,
    "sample_num": 100,

    "n_classes": 10
}

CELEBA_DEFAULTS = {
    "data_path": "/persist/datasets/celeba/img_align_celeba/all/",
    "model": "DeepConvResNet",
    "im_size": 64,
    "n_epochs": 1000,
    "g_lr": 0.0001,
    "d_lr": 0.0001,
    "batch_size": 128,
    "batch_split_size": 32,
    "train_set_size": 180000,
    "public_set_size": 0,
    "g_latent_dim": 128,
    "n_d_steps": 5,
    "adam_b1": 0.0,
    "adam_b2": 0.9,
    "penalty": ["WGAN-GP"],
    "iter_on_mean_samples": 0,
    "mean_sample_size": 300,
    "mean_sample_noise_std": 0.12,
    "delta": 1e-6,
    "sigma": 0.5,
    "imm_sens_scaling_vec": [20, 2, 15, 1.5, 10, 1.5, 10, 1, 30],
    "imm_sens_scaling_mode": "moving-avg-pl",
    "grad_clip_mode": "moving-avg-pl",
    "C": 200,
    "C_per_layer": [1000, 200, 1000, 100, 1000, 100, 1000, 5, 2500], # These model specific defaults should be handled elsewhere
    "adaptive_scalar": [2, 2, 2, 2, 2, 2, 2, 2, 2],
    "save_every": 10,
    "log_every": 20000,
    "sample_every": 60000,
    "sample_num": 25,

    "n_classes": 5,
    "gp_lambda": 10 # Gradient penalty
}

def fill_defaults(opt, default_dict):
    for key, val in default_dict.items():
        if not key in opt.__dict__ or opt.__dict__[key] is None or opt.__dict__[key] is False:
            opt.__dict__[key] = val

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manual_seed", type=int, default=-1)


    parser.add_argument("dataset", type=str, choices=["MNIST", "CelebA"])
    parser.add_argument("-d", "--data_path", type=str, default=None)
    parser.add_argument("--model", type=str, choices=["Vanilla", "DeepConvResNet"], default=None)
    parser.add_argument("--im_size", type=int, default=None, choices=[64, 48])
    parser.add_argument("--download_mnist", default=False, action="store_true")
    parser.add_argument("-o", "--output_dir", type=str, default=None) # will generate if None

    parser.add_argument("-rp", "--resume_path", type=str, default=None)
    parser.add_argument("-re", "--resume_epochs", type=int, default=0)
    parser.add_argument("-ka", "--keep_args", type=str, nargs="*", default=[])
    always_keep_args = ["g_device", "d_device", "num_workers", "resume_path", "resume_epochs"]

    parser.add_argument("-ne", "--n_epochs", type=int, default=None)
    parser.add_argument("--d_lr", type=float, default=None)
    parser.add_argument("--g_lr", type=float, default=None)
    parser.add_argument("-bs", "--batch_size", type=int, default=None)
    parser.add_argument("-bss", "--batch_split_size", type=int, default=None) # only for model parallel
    parser.add_argument("-tss", "--train_set_size", type=int, default=None)

    parser.add_argument("-gd", "--g_device", type=str, default="cpu")
    parser.add_argument("-dd", "--d_device", type=str, default="cpu")
    parser.add_argument("-nw", "--num_workers", type=int, default=8)

    parser.add_argument("--g_latent_dim", type=int, default=None)
    parser.add_argument("--n_d_steps", type=int, default=None)
    parser.add_argument("--conditional", type=bool, default=False)
    parser.add_argument("--adam_b1", type=float, default=None)
    parser.add_argument("--adam_b2", type=float, default=None)
    parser.add_argument("--penalty", type=str, nargs="*", choices=[None, "WGAN-GP", "WGAN-GP1", "DRAGAN", "DRAGAN1"], default=None, help="Specify a gradient penalty or list of gradient penalties. Names ending with a 1 indicate a one-sided penalty (only penalize being over the threshold and not also under).")

    parser.add_argument("-pss", "--public_set_size", type=int, default=0)
    parser.add_argument("-nms", "--num_mean_samples", type=int, default=0)
    parser.add_argument("-pupd", "--penalty_use_public_data", type=bool, default=True)
    parser.add_argument("-wi", "--warmup_iter", type=int, default=0)

    parser.add_argument("--mean_sample_size", type=int, default=None)
    parser.add_argument("--mean_sample_noise_std", type=int, default=None)

    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("-eb", "--epsilon_budget", type=float, default=None)

    parser.add_argument("-uis", "--use_imm_sens", action="store_true", default=False)
    parser.add_argument("-ispp", "--imm_sens_per_param", type=bool, default=False, help="Calculates IS for each parameter separately.")
    parser.add_argument("-issv", "--imm_sens_scaling_vec", type=float, nargs="*", default=None)
    parser.add_argument("-issm", "--imm_sens_scaling_mode", type=str, choices=[None, "constant-pl", "moving-avg-pl"], default=None,
        help="constant per-layer, moving avg per-layer (updates v = v*beta + grad_norm*(1-beta) per layer)")

    parser.add_argument("-ugc", "--use_grad_clip", action="store_true", default=False)
    parser.add_argument("-gcs", "--grad_clip_split", type=bool, default=True)
    parser.add_argument("-gcm", "--grad_clip_mode", type=str, choices=["standard", "constant-pl", "adaptive-pl", "moving-avg-pl"], default=None,
        help="Gradient clipping mode: standard (clips overall grad norm), constant per-layer, adaptive per-layer (uses either public partition of data or public mean samples and scales adaptive_stat of the data by adaptive_scalar per layer), moving avg per-layer (updates v = v*beta + grad_norm*target_scale*(1-beta) per layer)")
    parser.add_argument("-c", "--clipping_param", type=float, default=None)
    parser.add_argument("-cpl", "--clipping_param_per_layer", type=float, nargs="*", default=None)
    parser.add_argument("-as", "--adaptive_scalar", type=float, nargs="*", default=None)
    parser.add_argument("--adaptive_stat", choices=["mean", "max"], default="mean")
    parser.add_argument("-mat", "--moving_avg_target", type=float, default=2.5) # Only for GC (value for IS is just 1)
    parser.add_argument("-mab", "--moving_avg_beta", type=float, default=0.98) # Applies to both GC and IS

    parser.add_argument("--save_every", type=int, default=None) # epochs
    parser.add_argument("--log_every", type=int, default=None) # samples, prints and logs to csv
    parser.add_argument("--sample_every", type=int, default=None) # samples
    parser.add_argument("--sample_num", type=int, default=None)
    parser.add_argument("-p", "--profile_training", default=False, action="store_true")

    opt = parser.parse_args()
    opt.keep_args = opt.keep_args + always_keep_args

    # Add slash to end of paths if needed
    opt.data_path = util.add_slash(opt.data_path)
    opt.resume_path = util.add_slash(opt.resume_path)
    opt.output_dir = util.add_slash(opt.output_dir)

    if opt.resume_path is None:
        if opt.dataset == "MNIST":
            fill_defaults(opt, MNIST_DEFAULTS)
        elif opt.dataset == "CelebA":
            fill_defaults(opt, CELEBA_DEFAULTS)

        # Set logging in terms of epochs if long enough
        opt.log_every_epochs = -1 if opt.log_every < opt.train_set_size else opt.log_every // opt.train_set_size
        opt.sample_every_epochs = -1 if opt.sample_every < opt.train_set_size else opt.sample_every // opt.train_set_size

        # Correct intervals to be multiples of batch size
        opt.log_every = max((opt.log_every // opt.batch_size)*opt.batch_size, 1)
        opt.sample_every = max((opt.sample_every // opt.batch_size)*opt.batch_size, 1)

        opt.use_dp = opt.use_grad_clip or opt.use_imm_sens
        opt.use_grad_clip_per_layer = opt.grad_clip_mode != "standard"

        # Check for incompatible configurations
        if opt.imm_sens_per_param and not opt.imm_sens_scaling_mode is None:
            raise Exception("Calculating IS per parameter does not require per parameter scaling. Scaling estimates per-parameter calculation.")
        if opt.imm_sens_per_param:
            print("Not sure if immediate sensitivity per parameter is working properly.")
        if opt.public_set_size > 0 and opt.num_mean_samples > 0:
            raise Exception("Both public data partition and mean samples were configured, please select only one.")
        if len(opt.penalty) > 0 and opt.use_grad_clip and opt.penalty_use_public_data and opt.public_set_size < 1 and opt.num_mean_samples < 1:
            raise Exception("In order to enable gradient penalty using public data, please enable mean sampling by setting num_mean_samples or public data by setting public_set_size.")
        if len(opt.penalty) > 0 and opt.use_grad_clip and opt.public_set_size < 1 and opt.num_mean_samples < 1:
            print("Currently configured to calculate penalty per-sample. It is strongly recommended that you use public data or mean samples for gradient penalties when using grad clipping.")
        if opt.conditional and not (opt.penalty is None or len(opt.penalty) < 1):
            raise Exception("Penalties not yet implemented for conditional architectures.")
        if opt.conditional and opt.iter_on_mean_samples > 0:
            raise Exception("Mean sampling not yet implemented for conditional architectures.")

        # Generate output directory if not specified
        if opt.output_dir == None or opt.output_dir == "":
            now = datetime.now()
            opt.output_dir = now.strftime("output/%m-%d-%H:%M-") + opt.dataset + "-g" + str(opt.g_device)[-1] + "-d" + str(opt.d_device)[-1] + "/"
        for path in ["output", opt.output_dir, opt.output_dir+"samples/", opt.output_dir+"saves/", opt.output_dir+"code/"]:
            if not os.path.exists(path):
                os.makedirs(path)

        # Generate seed
        if opt.manual_seed < 0:
            opt.manual_seed = random.randint(1, 1000000)
        random.seed(opt.manual_seed)
        torch.manual_seed(opt.manual_seed)
    else:
        # Load options if resuming
        loaded_opt = load_opt(opt.resume_path + "opt.txt")

        for arg in opt.keep_args:
            setattr(loaded_opt, arg, getattr(opt, arg))
        opt = loaded_opt

        opt.output_dir = opt.resume_path

    return opt

def load_opt(path):
    opt = Namespace()
    with open(path, "r") as f:
        opt.__dict__ = json.load(f)
    return opt
