import torch
import torchvision
import glob, os
import numpy as np
from opacus.privacy_analysis import compute_rdp, get_privacy_spent

import util

ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


class MeanSampler():
    def __init__(self, dataloader=None, path=None, transforms=None, noise_std=0.1, num_samples=32, mean_size=100, dataset_size=180000, res=64, ch=3, save_path=None, default_batch_size=None):
        self.dataloader = dataloader
        self.noise_std = noise_std
        self.num_samples = num_samples
        self.mean_size = mean_size
        self.dataset_size = dataset_size
        self.res = res
        self.ch = ch
        self.default_batch_size=default_batch_size

        self.sample_rate = self.mean_size/self.dataset_size

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
        self.mean_samples = torch.tensor([])
        save_path = util.add_slash(save_path)

        for _ in range(self.num_samples):
            samples, _ = next(iter(dataloader))
            
            mean_sample = samples.sum(dim=0) / self.mean_size # Divide by mean_size and sum to ensure sensitivity bound for each sample
            mean_sample += torch.empty(mean_sample.shape).normal_(0, self.noise_std)
            self.mean_samples = torch.cat((self.mean_samples, torch.unsqueeze(mean_sample, dim=0)), dim=0)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not save_path is None:
            for i in range(self.mean_samples.size(0)):
                torchvision.utils.save_image(util.denorm_celeba(self.mean_samples[i].data), save_path+str(i+1)+".png", nrow=1)

    def sample(self, size, noise_std=0.04, noise_mean_std=0.04):
        perm = torch.randperm(self.num_samples)
        # Repeat perm when size > num_samples
        n_repeats = (size-1) // self.num_samples + 1
        perm = torch.cat([perm for _ in range(n_repeats)], dim=0)

        idx = perm[:size]
        r = self.mean_samples[idx]
        if not noise_mean_std is None and noise_mean_std > 0:
            r += torch.empty(r.size(0)).normal_(0, noise_mean_std).view(-1, 1, 1, 1).expand(*r.shape)
        if not noise_std is None and noise_std > 0:
            r += torch.empty(r.shape).normal_(0, noise_std)
        return r

    def get_privacy_cost(self, target_delta=1e-6, alphas=ALPHAS):
        pixel_sensitivity = 1/self.mean_size
        l2_sensitivity = np.sqrt(self.ch*self.res**2*pixel_sensitivity**2)

        # Opacus accounting
        rdp = compute_rdp(self.sample_rate, self.noise_std / l2_sensitivity, self.num_samples, orders=alphas)
        return get_privacy_spent(orders=alphas, rdp=rdp, delta=target_delta)
