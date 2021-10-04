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
    public_dataset = None
    if opt.dataset == "MNIST":
        dataset = datasets.MNIST(root=opt.data_path, train=True, download=opt.download_mnist, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
        if opt.public_set_size > 0:
            dataset, public_dataset = torch.utils.data.random_split(ds, [opt.train_set_size, opt.public_set_size])
    elif opt.dataset == "CelebA":
        dataset = CelebADataset(opt.data_path, im_size=opt.im_size, length=opt.train_set_size)
        if opt.public_set_size > 0:
            public_dataset = CelebADataset(opt.data_path, im_size=opt.im_size, length=opt.public_set_size, offset=opt.train_set_size)

    dataloader = DataLoader(dataset=dataset, batch_sampler=UniformWithReplacementSampler(
        num_samples=opt.train_set_size,
        sample_rate=opt.batch_size/opt.train_set_size
    ), num_workers=opt.num_workers)
    public_dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True) if opt.public_set_size > 0 else None

    return dataset, dataloader, public_dataset, public_dataloader

def init_models(opt, init_G=True, init_D=True):
    n_classes = opt.n_classes if opt.conditional else 0
    bn = not opt.use_grad_clip
    G, D = None, None

    if opt.dataset == "MNIST":
        if opt.model == "DeepConvResNet":
            G = MNIST_DCRN_G(opt.g_latent_dim, bn=bn, n_classes=n_classes).to(opt.g_device) if init_G else None
            D = MNIST_DCRN_D(n_classes=n_classes).to(opt.d_device) if init_D else None
        elif opt.model == "Vanilla":
            G = MNISTVanillaG(opt.g_latent_dim, n_classes=n_classes).to(opt.g_device) if init_G else None
            D = MNISTVanillaD(n_classes=n_classes).to(opt.d_device) if init_D else None
    elif opt.dataset == "CelebA":
        if opt.model == "DeepConvResNet":
            GObj = CelebA_DCRN_G48 if opt.im_size == 48 else CelebA_DCRN_G64
            DObj = CelebA_DCRN_D48 if opt.im_size == 48 else CelebA_DCRN_D64
            G = GObj(z_dim=opt.g_latent_dim, bn=bn, n_classes=n_classes).to(opt.g_device) if init_G else None
            D = DObj(n_classes=n_classes).to(opt.d_device) if init_D else None
        elif opt.model == "Vanilla":
            raise Exception("No vanilla architecture for CelebA.")

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
