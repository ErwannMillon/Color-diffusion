import glob
import random
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageChops

from utils import load_default_configs, split_lab_channels


def is_greyscale(im):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    You can use this to filter your dataset of black and white images 
    """
    if isinstance(im, str):
        im = Image.open(im).convert("RGB")
    if im.mode not in ("L", "RGB"):
        raise ValueError("Unsuported image mode")
    if im.mode == "RGB":
        rgb = im.split()
        if ImageChops.difference(rgb[0], rgb[1]).getextrema()[1] != 0:
            return False
        if ImageChops.difference(rgb[0], rgb[2]).getextrema()[1] != 0:
            return False
    return True


class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train', config=None):
        size = config["img_size"]
        self.resize = transforms.Resize((size, size), Image.BICUBIC)
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3,
                                       contrast=0.1,
                                       saturation=(1., 2.),
                                       hue=0.05),
                self.resize
            ])
        elif split == 'val':
            self.transforms = self.resize
        self.paths = paths

    def tensor_to_lab(self, base_img_tens):
        base_img = np.array(base_img_tens)
        img_lab = rgb2lab(base_img).astype(
            "float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
        return torch.cat((L, ab), dim=0)

    def get_lab_from_path(self, path):
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)
        return self.tensor_to_lab(img)

    def get_rgb(self, idx=0):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        return (img)

    def get_grayscale(self, idx=0):
        img = Image.open(self.paths[idx]).convert("L")
        img = self.resize(img)
        img = np.array(img)
        return (img)

    def get_lab_grayscale(self, idx=0):
        img = self.get_lab_from_path(self.paths[idx])
        l, _ = split_lab_channels(img.unsqueeze(0))
        return torch.cat((l, *[torch.zeros_like(l)] * 2), dim=1)

    def __getitem__(self, idx):
        return self.get_lab_from_path(self.paths[idx])

    def __len__(self):
        return len(self.paths)


class PickleColorizationDataset(ColorizationDataset):
    def __getitem__(self, idx):
        return (torch.load(self.paths[idx]))

def make_datasets(path, config, limit=None):
    img_paths = glob.glob(path + "/*")
    if limit:
        img_paths = random.sample(img_paths, limit)
    n_imgs = len(img_paths)
    train_split = img_paths[:int(n_imgs * .9)]
    val_split = img_paths[int(n_imgs * .9):]

    train_dataset = ColorizationDataset(
        train_split, split="train", config=config)
    val_dataset = ColorizationDataset(val_split, split="val", config=config)
    print(f"Train size: {len(train_split)}")
    print(f"Val size: {len(val_split)}")
    return train_dataset, val_dataset


def make_dataloaders(path, config, num_workers=2, shuffle=True, limit=None):
    train_dataset, val_dataset = make_datasets(path, config, limit=limit)
    train_dl = DataLoader(train_dataset,
                          batch_size=config["batch_size"],
                          num_workers=num_workers,
                          pin_memory=config["pin_memory"],
                          persistent_workers=True,
                          shuffle=shuffle)
    val_dl = DataLoader(val_dataset,
                        batch_size=config["batch_size"],
                        num_workers=num_workers,
                        pin_memory=config["pin_memory"],
                        persistent_workers=True,
                        shuffle=shuffle)
    return train_dl, val_dl


if __name__ == "__main__":
    enc_config, unet_config, colordiff_config = load_default_configs()
    train_dl, val_dl = make_dataloaders("./fairface",
                                        colordiff_config,
                                        num_workers=4)
    x = next(iter(train_dl))
    y = next(iter(val_dl))
    print(f"y.shape = {y.shape}")
    print(f"x.shape = {x.shape}")
