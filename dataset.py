import glob
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train', size=64, config=None, limit=None):
        if config:
            size = config["img_size"]
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((size, size), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),  # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((size, size), Image.BICUBIC)

        self.split = split
        self.size = size
        self.paths = paths[:limit]

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
        return (torch.cat((L, ab), dim=0))
        return {'L': L, 'ab': ab}
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

def make_dataloaders(path, config, num_workers=0, limit=None):
    train_paths = glob.glob(path + "/train/*.jpg")
    train_dataset = ColorizationDataset(train_paths, split="train", size=config["img_size"], limit=limit)
    train_dl = DataLoader(train_dataset, batch_size=config["batch_size"], 
                            num_workers=num_workers, pin_memory=config["pin_memory"])
    val_paths = glob.glob(path + "/val/*.jpg")
    val_dataset = ColorizationDataset(val_paths, split="val", size=config["img_size"], limit=limit)
    val_dl = DataLoader(val_dataset, batch_size=config["batch_size"], 
                            num_workers=num_workers, pin_memory=config["pin_memory"], shuffle=True)
    return train_dl, val_dl
