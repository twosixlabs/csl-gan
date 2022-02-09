import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import autograd
from torchvision import datasets
import torch.nn.functional as F

import util
from datasets import *
from MNIST_models import *
from CelebA_models import *

def init_data(opt):
    dataset, public_dataset = None, None
    if opt.dataset == "MNIST":
        dataset = datasets.MNIST(root=opt.data_path, train=True, download=opt.download_mnist, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
        stratified_dataset = []
        for label in range(10):
            label_samples = [(x, y) for (i, (x, y)) in enumerate(dataset) if y == label]
            stratified_dataset.extend(label_samples[:opt.train_set_size // 10])
        dataset = stratified_dataset
        if opt.public_set_size > 0:
            public_dataset = datasets.MNIST(root=opt.data_path, train=False, download=opt.download_mnist, transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
            #dss, pdss = torch.utils.data.random_split(dataset, [opt.train_set_size, opt.public_set_size])
            #dataset = dss.dataset
            #public_dataset = pdss.dataset

    elif opt.dataset == "CelebA":
        dataset = CelebADataset(opt.data_path, im_size=opt.im_size, length=opt.train_set_size, attr_file=opt.label_path, attr=opt.label_attr)
        if opt.public_set_size > 0:
            public_dataset = CelebADataset(opt.data_path, im_size=opt.im_size, length=opt.public_set_size, offset=opt.train_set_size, attr_file=opt.label_path, attr=opt.label_attr)

    #dataloader = DataLoader(dataset=dataset, num_workers=opt.num_workers, pin_memory=False, batch_size=opt.batch_size,)
    #public_dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True) if opt.public_set_size > 0 else None
    dataloader = DataLoader(dataset=dataset, num_workers=opt.num_workers, pin_memory=False, batch_size=opt.batch_size, shuffle=True)
    public_dataloader = DataLoader(dataset=public_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True) if opt.public_set_size > 0 else None

    return dataset, dataloader, public_dataset, public_dataloader

def init_models(opt, init_G=True, init_D=True):
    n_classes = opt.n_classes if opt.conditional else 0
    bn = not opt.per_sample_grad
    GObj, DObj = None, None

    if opt.dataset == "MNIST":
        if opt.model == "DeepConvResNet":
            GObj = MNIST_DCRN_G
            DObj = MNIST_DCRN_D
        elif opt.model == "Vanilla":
            GObj = MNISTVanillaG
            DObj = MNISTVanillaD
    elif opt.dataset == "CelebA":
        if opt.model == "DeepConvResNet":
            GObj = CelebA_DCRN_G48 if opt.im_size == 48 else CelebA_DCRN_G64
            DObj = CelebA_DCRN_D48 if opt.im_size == 48 else CelebA_DCRN_D64
        elif opt.model == "Vanilla":
            raise Exception("No vanilla architecture for CelebA.")

    torch.manual_seed(opt.weights_seed)
    torch.cuda.manual_seed(opt.weights_seed)
    G = GObj(z_dim=opt.g_latent_dim, bn=bn, n_classes=n_classes, emb_mode=opt.g_label_emb_mode).to(opt.g_device) if init_G else None
    D = DObj(n_classes=n_classes, emb_mode=opt.d_label_emb_mode, conditional_arch=opt.conditional_arch,
            aux_loss_type=opt.aux_loss_type, aux_loss_scalar=opt.aux_loss_scalar).to(opt.d_device) if init_D else None
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    return G, D
