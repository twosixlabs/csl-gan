import os
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import pandas as pd

CELEBA_ATTR = ["Filename", "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
       "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
       "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
       "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
       "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
       "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
       "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
       "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
       "Wearing_Necklace", "Wearing_Necktie", "Young"]

class CelebADataset(VisionDataset):
    def __init__(self, root, im_size=32, length=None, offset=0, ext="jpg", attr_file=None, attr=None):

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

        if attr is None:
            self.labels = None
            self.label_true_count = None
        else:
            self.labels = (pd.read_csv(attr_file, sep=" ", skiprows=2, header=None, names=CELEBA_ATTR).head(self.length + self.offset).tail(self.length)[attr].values == 1).astype(int)
            self.label_true_count = 0 if attr is None else (self.labels == 1).sum()

        self.n_classes = 1

    def __len__(self):
        return self.length

    def getSample(self, number):
        file = str(self.offset + number).zfill(6) + "." + self.ext
        image_path = os.path.join(self.root, file )
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)

        return img, self.labels[number-1]

    def __getitem__(self, index):
        return self.getSample(index+1)

    def get_item_with_label(self, label, number=None):
        number = np.random.randint(0, self.length) if number is None else number
        while self.labels[number] != label:
            number  = (number + 1) % self.length
        return self.__getitem__(number)
