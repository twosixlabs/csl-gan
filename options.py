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
    "im_size": 28,
    "n_epochs": 10000,
    "g_lr": 0.00002,
    "d_lr": 0.00002,
    "batch_size": 600,
    "batch_split_size": 60,
    "train_set_size": 60000,
    "g_latent_dim": 100,
    "n_d_steps": 1,
    "unconditional": False,
    "adam_b1": 0.9,
    "adam_b2": 0.999,
    "penalty_use_mean_samples": False,
    "penalty": [],
    "iter_on_mean_samples": 0,
    "delta": 1e-5,
    "sigma": 5.0,
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
    "im_size": 64,
    "n_epochs": 1000,
    "g_lr": 0.0001,
    "d_lr": 0.0001,
    "batch_size": 128,
    "batch_split_size": 32,
    "train_set_size": 180000,
    "g_latent_dim": 128,
    "n_d_steps": 5,
    "unconditional": True,
    "adam_b1": 0.0,
    "adam_b2": 0.9,
    "penalty": ["WGAN-GP"],
    "penalty_use_mean_samples": False,
    "iter_on_mean_samples": 0,
    "mean_sample_size": 200,
    "num_mean_samples": 80,
    "mean_sample_noise_std": 0.1,
    "delta": 1e-6,
    "sigma": 0.2,
    "C": 30.0,
    "C_weight": 1000.0,
    "C_bias": 100.0,
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
    parser.add_argument("--unconditional", action="store_true", default=False)
    parser.add_argument("--adam_b1", type=float, default=None)
    parser.add_argument("--adam_b2", type=float, default=None)
    parser.add_argument("--penalty", type=str, nargs="*", choices=["WGAN-GP", "WGAN-GP1", "DRAGAN", "DRAGAN1"], default=None)

    parser.add_argument("-pums", "--penalty_use_mean_samples", action="store_true", default=False)
    parser.add_argument("-ioms", "--iter_on_mean_samples", type=int, default=None)
    parser.add_argument("--mean_sample_size", type=int, default=None)
    parser.add_argument("--num_mean_samples", type=int, default=None)
    parser.add_argument("--mean_sample_noise_std", type=int, default=None)

    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("-is", "--use_imm_sens", action="store_true", default=False)
    parser.add_argument("-ppis", "--use_per_param_imm_sens", action="store_true", default=False)
    parser.add_argument("-plc", "--use_per_layer_clipping", action="store_true", default=False)
    parser.add_argument("-sgc", "--use_split_grad_clip", action="store_true", default=False)
    parser.add_argument("-gc", "--use_grad_clip", action="store_true", default=False)
    parser.add_argument("--C", type=float, default=None)
    parser.add_argument("--C_weight", type=float, default=None)
    parser.add_argument("--C_bias", type=float, default=None)
    parser.add_argument("-eb", "--epsilon_budget", type=float, default=None)

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

        opt.use_dp_per_sample = opt.use_grad_clip or opt.use_split_grad_clip or opt.use_per_sample_imm_sens
        opt.use_dp = opt.use_dp_per_sample or opt.use_imm_sens

        # Generate output directory if not specified
        if opt.output_dir == None or output_dir == "":
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

        if not opt.unconditional and not (opt.penalty is None or len(opt.penalty) < 1):
            raise Exception("Penalties not yet implemented for conditional architectures.")
        if not opt.unconditional and opt.iter_on_mean_samples > 0:
            raise Exception("Mean sampling not yet implemented for conditional architectures.")
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
