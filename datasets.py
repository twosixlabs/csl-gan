import os
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

class CelebADataset(VisionDataset):
    def __init__(self, root, im_size=32, length=None, offset=0, ext="jpg"):

        self.root = root
        all_files = os.listdir(self.root)
        self.length = length if length else len(all_files)
        self.offset = offset
        self.ext = ext
        self.transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.n_classes = 1

    def __len__(self):
        return self.length

    def getImage(self, number):
        file = str(self.offset + number).zfill(6) + "." + self.ext
        image_path = os.path.join(self.root, file )
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)

        return img

    def __getitem__(self, index):
        return self.getImage(index+1), 0