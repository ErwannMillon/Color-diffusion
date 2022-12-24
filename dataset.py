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
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=(1., 1.8), hue=0.05),
                # transforms.GaussianBlur(kernel_size=3, sigma=(0.5, .5)),
                transforms.Resize((size, size), Image.BICUBIC)
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((size, size), Image.BICUBIC)
        self.device = config["device"]
        self.split = split
        self.size = size
        self.paths = paths[:limit]
    def get_tensor_from_path(self, path):
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
        return (torch.cat((L, ab), dim=0))
    def __getitem__(self, idx):
        x = self.get_tensor_from_path(self.paths[idx])
        return x
    def tensor_to_lab(self, base_img_tens):
        base_img = np.array(base_img_tens)
        img_lab = rgb2lab(base_img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
        return torch.cat((L, ab), dim=0)
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

class PickleColorizationDataset(ColorizationDataset):
    def __getitem__(self, idx):
        return(torch.load(self.paths[idx]))
import csv

def make_dataloaders_celeba(path, config, num_workers=0, shuffle=True, limit=None):
    img_paths = glob.glob(path + "/*")
    n_imgs = len(img_paths)
    train_split = img_paths[:int(n_imgs * .9)]
    val_split = img_paths[int(n_imgs * .9):]
    train_dataset = ColorizationDataset(train_split, split="train", config=config, size=config["img_size"], limit=limit)
    val_dataset = ColorizationDataset(val_split, split="val", config=config, size=config["img_size"], limit=limit//10)
    print(f"train size: {len(train_split)}")
    train_dl = DataLoader(train_dataset, batch_size=config["batch_size"], 
                            num_workers=num_workers, pin_memory=config["pin_memory"], persistent_workers=True, shuffle=shuffle)
    print(f"val size: {len(val_split)}")
    val_dl = DataLoader(val_dataset, batch_size=config["batch_size"], 
                            num_workers=num_workers, pin_memory=config["pin_memory"], persistent_workers=True, shuffle=shuffle)
    return train_dl, val_dl

def make_dataloaders(path, config, use_csv=True, num_workers=0, limit=None, pickle=True):
    if pickle:
        use_csv = False
    train_paths = glob.glob(path + "/train/*")
    val_paths = glob.glob(path + "/val/*")
    if use_csv:
        with open("./train_filtered.csv", "r") as f:
            reader = csv.reader(f)
            data = list(reader)
            train_paths = data[0]
        with open("./val_filtered.csv", "r") as f:
            reader = csv.reader(f)
            data = list(reader)
            val_paths = data[0]
    if pickle:
        train_dataset = PickleColorizationDataset(train_paths, split="train", config=config, size=config["img_size"], limit=limit)
        val_dataset = PickleColorizationDataset(val_paths, split="val", config=config, size=config["img_size"], limit=limit)
    else:
        train_dataset = ColorizationDataset(train_paths, split="train", config=config, size=config["img_size"], limit=limit)
        val_dataset = ColorizationDataset(val_paths, split="val", config=config, size=config["img_size"], limit=limit)

    print(f"train size: {len(train_dataset.paths)}")
    train_dl = DataLoader(train_dataset, batch_size=config["batch_size"], 
                            num_workers=num_workers, pin_memory=config["pin_memory"])
    print(f"val size: {len(val_dataset.paths)}")
    val_dl = DataLoader(val_dataset, batch_size=config["batch_size"], 
                            num_workers=num_workers, pin_memory=config["pin_memory"], shuffle=shuffle)
    return train_dl, val_dl
if __name__ == "__main__":
    from default_configs import ColorDiffConfig
    config = ColorDiffConfig
    train_dl, val_dl = make_dataloaders("./fairface", config, pickle=False, use_csv=True, num_workers=4)
    x=next(iter(train_dl))
    y=next(iter(val_dl))
    print(f"y.shape = {y.shape}")
    print(f"x.shape = {x.shape}")