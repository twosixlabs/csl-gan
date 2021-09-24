import torch
from torchvision import transforms
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data import DataLoader
from torch import autograd
from torchvision import datasets

from datasets import *
from MNIST_models import *
from CelebA_models import *


def add_slash(path):
    return None if path is None else (path if path[-1] == "/" else path + "/")

def denorm_celeba(img):
    return ((img + 1) / 2).clamp(0, 1)

def save_model(epoch, model, optimizer, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, path)

def init_data(opt):
    dataset = None
    if opt.dataset == "MNIST":
        dataset = datasets.MNIST(root=opt.data_path, train=True, download=opt.download_mnist, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
    elif opt.dataset == "CelebA":
        dataset = CelebADataset(opt.data_path, im_size=opt.im_size)

    dataloader = DataLoader(dataset=dataset, batch_sampler=UniformWithReplacementSampler(
        num_samples=opt.train_set_size,
        sample_rate=opt.batch_size/opt.train_set_size
    ), num_workers=opt.num_workers)

    return dataset, dataloader

def init_models(opt, init_G=True, init_D=True):
    if opt.dataset == "MNIST":
        G = MNISTVanillaG(opt.g_latent_dim, unconditional=opt.unconditional).to(opt.g_device) if init_G else None
        D = MNISTVanillaD(unconditional=opt.unconditional).to(opt.d_device) if init_D else None
    elif opt.dataset == "CelebA":
        dataset = CelebADataset(opt.data_path, im_size=opt.im_size)
        GObj = ResNetDCGenerator48 if opt.im_size == 48 else ResNetDCGenerator64
        DObj = ResNetDCDiscriminator48 if opt.im_size == 48 else ResNetDCDiscriminator64
        G = GObj(z_dim=opt.g_latent_dim, bn=not (opt.use_grad_clip or opt.use_split_grad_clip)).to(opt.g_device) if init_G else None
        D = DObj(gp_lambda=opt.gp_lambda).to(opt.d_device) if init_D else None

    return G, D

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)

def load_model(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if not optimizer is None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"]
