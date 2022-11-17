import glob
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageChops

from PIL import Image, ImageStat

def is_alt_greyscale(path="image.jpg"):
    im = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(im)
    if sum(stat.sum)/3 == stat.sum[0]: #check the avg with any element value
        return True #if grayscale
    else:
        return False #else its colour

def is_greyscale(im):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    """
    if isinstance(im, str):
        im = Image.open(im).convert("RGB")
    if im.mode not in ("L", "RGB"):
        raise ValueError("Unsuported image mode")
    if im.mode == "RGB":
        rgb = im.split()
        if ImageChops.difference(rgb[0],rgb[1]).getextrema()[1]!=0: 
            return False
        if ImageChops.difference(rgb[0],rgb[2]).getextrema()[1]!=0: 
            return False
    return True
class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train', size=64, config=None, limit=None):
        if config:
            size = config["img_size"]
        if split == 'train':
            self.transforms = transforms.Compose([
            #    transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.45, hue=0.02),
                transforms.RandomHorizontalFlip(),  # A little data augmentation!
                transforms.GaussianBlur(kernel_size=3, sigma=(0.5, .5)),
                transforms.Resize((size, size), Image.BICUBIC)
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((size, size), Image.BICUBIC)

        self.split = split
        self.size = size
        self.paths = paths[:limit]
        self.paths = [path for path in self.paths if not is_alt_greyscale(path)]

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        # while (is_greyscale(img) is True):
        #     idx
        #     self.paths.pop(idx)
        #     img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
        return (torch.cat((L, ab), dim=0))
    def tensor_to_lab(self, base_img_tens):
        base_img = np.array(base_img_tens)
        img_lab = rgb2lab(base_img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
        return torch.cat((L, ab), dim=0)
    def test(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        base_img_tens = self.transforms(img)
        base_img = np.array(base_img_tens)
        img_lab = rgb2lab(base_img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
        base_lab = torch.cat((L, ab), dim=0)
        col = transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.6, hue=0.02)
        blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.5, .5))
        trans_col = self.tensor_to_lab(col(base_img_tens))
        trans_blur = self.tensor_to_lab(blur(base_img_tens))
        trans_both = self.tensor_to_lab(col(blur(base_img_tens)))
        return (base_lab, trans_col, trans_blur, trans_both)

    def get_rgb(self):
        img = Image.open(self.paths[0]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        return (img)
    def get_grayscale(self):
        img = Image.open(self.paths[0]).convert("L")
        img = self.transforms(img)
        img = np.array(img)
        return (img)
    def __len__(self):
        return len(self.paths)

import csv
def make_dataloaders(path, config, num_workers=0, limit=None):
    # train_paths = glob.glob(path + "/train/*.jpg")
    # val_paths = glob.glob(path + "/val/*.jpg")
    with open("./train_filtered.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        train_paths = data[0]
    with open("./val_filtered.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        val_paths = data[0]
    train_dataset = ColorizationDataset(train_paths, split="train", size=config["img_size"], limit=limit)
    train_dl = DataLoader(train_dataset, batch_size=config["batch_size"], 
                            num_workers=num_workers, pin_memory=config["pin_memory"])
    val_dataset = ColorizationDataset(val_paths, split="val", size=config["img_size"], limit=limit)
    val_dl = DataLoader(val_dataset, batch_size=config["batch_size"], 
                            num_workers=num_workers, pin_memory=config["pin_memory"], shuffle=True)
    return train_dl, val_dl
