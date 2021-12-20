import torch
from torchvision import transforms
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data import DataLoader
from torch import autograd
from torchvision import datasets
import torch.nn.functional as F


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

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)

def zero_grad(model):
    for p in model.parameters():
        p.grad = None

def load_model(path, model, device, optimizer=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if not optimizer is None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"]

def max_batch_size_for_sample_rate(sample_rate, train_set_size=180000, acceptable_risk_per_batch=0.002):
    # Acceptable risk is probability of returned batch size being exceeded
    # Normal approximation of binomial distribution
    mu = train_set_size*sample_rate
    sigma = np.sqrt(train_set_size*sample_rate*(1-sample_rate))
    dist = torch.distributions.normal.Normal(mu, sigma)
    return int(dist.icdf(torch.tensor(1 - acceptable_risk_per_batch)).item()) + 1

def convert_modules(self, module_type, replacement):
    for name, m in module.named_modules():
        if isinstance(m, module_type) and len(list(m.children())) < 1:
            parent = module
            names = name.split(".")
            for name in names[:-1]:
                parent = parent._modules[name]

            parent._modules[names[-1]] = replacement(self, m)
