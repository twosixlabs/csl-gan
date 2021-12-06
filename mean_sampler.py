import torch
import torchvision
import glob, os
import numpy as np
from opacus.privacy_analysis import compute_rdp, get_privacy_spent

import util

ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


class MeanSampler():
    def __init__(self, dataloader=None, path=None, transforms=None, noise_std=0.1, num_samples=32,
                mean_size=100, dataset_size=180000, res=64, ch=3, save_path=None, default_batch_size=None,
                n_classes=1, smallest_class_size=None):
        # mean_size and num _samples are per-class
        self.dataloader = dataloader
        self.noise_std = noise_std
        self.num_samples = num_samples
        self.mean_size = mean_size
        self.dataset_size = dataset_size
        self.res = res
        self.ch = ch
        self.default_batch_size=default_batch_size

        self.sample_rate = self.mean_size/self.dataset_size if smallest_class_size is None else self.mean_size/smallest_class_size
        self.smallest_class_size = smallest_class_size
        self.n_classes = n_classes

        if not path is None:
            self.load_mean_samples(path, transforms=transforms)
        elif not dataloader is None:
            self.make_mean_samples(dataloader, save_path=save_path)

    def load_mean_samples(self, path, transforms):
        jpg_files = glob.glob(path + "*.jpg")
        png_files = glob.glob(path + "*.png")

        self.mean_samples = []
        for file in jpg_files + png_files:
            img = Image.open(file).convert("RGB")
            if not transforms is None:
                img = transforms(img)

            self.mean_samples.append(img)
        self.mean_samples = torch.tensor(self.mean_samples)

    def make_mean_samples(self, dataloader, save_path=None):
        self.mean_samples = [[] for _ in range(self.n_classes)]
        save_path = util.add_slash(save_path)

        for i in range(self.num_samples):
            samples, labels = next(iter(dataloader))

            for c in range(self.n_classes):
                s = None
                if self.n_classes > 1:
                    ind = labels == c
                    s = samples[ind, :, :, :] # Cut off mean at mean_size to account for imbalanced classes
                    s = (s[:self.mean_size] if len(s) > self.mean_size else s).sum(dim=0) / self.mean_size
                else:
                    s = samples.sum(dim=0) / self.mean_size

                self.mean_samples[c].append(s + torch.empty(s.shape).normal_(0, self.noise_std))

        self.mean_samples = torch.stack([torch.stack(samples) for samples in self.mean_samples])

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not save_path is None:
            for c in range(self.mean_samples.size(0)):
                for i in range(self.mean_samples.size(1)):
                    torchvision.utils.save_image(util.denorm_celeba(self.mean_samples[c,i].data), save_path+str(c)+"-"+str(i+1)+".png", nrow=1)

    def sample(self, size, noise_std=0.01, noise_mean_std=0.01, requested_labels=None):
        perms = torch.cat([torch.randperm(self.num_samples) for _ in range((size-1)//self.num_samples+1)], dim=0)[:size]
        requested_labels = torch.empty((size)).random_(0, self.n_classes).long() if requested_labels is None else requested_labels

        r = self.mean_samples[requested_labels, perms]
        if not noise_mean_std is None and noise_mean_std > 0:
            r += torch.empty(r.size(0)).normal_(0, noise_mean_std).view(-1, 1, 1, 1).expand(*r.shape)
        if not noise_std is None and noise_std > 0:
            r += torch.empty(r.shape).normal_(0, noise_std)
        return r, (requested_labels if self.n_classes > 1 else None)

    def get_privacy_cost(self, target_delta=1e-6, alphas=ALPHAS):
        pixel_sensitivity = self.n_classes/self.mean_size
        l2_sensitivity = np.sqrt(self.ch*self.res**2*pixel_sensitivity**2)

        # Opacus accounting
        rdp = compute_rdp(self.sample_rate, self.noise_std / l2_sensitivity, self.num_samples * self.n_classes, orders=alphas)
        return get_privacy_spent(orders=alphas, rdp=rdp, delta=target_delta)
