import glob
import os
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.notebook import tqdm

from dataset import ColorizationDataset, make_dataloaders
from diffusion import forward_diffusion_sample
from utils import lab_to_rgb, show_lab_image, split_lab, visualize
from train import config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_colab = None
torch.manual_seed(0)

# dataset = ColorizationDataset(["./data/test.jpg"]);
# dataloader = DataLoader(dataset, batch_size=1)

train_dl, val_dl = make_dataloaders("./fairface", config)
x = next(iter(train_dl))
show_lab_image(x, log=False)
plt.show()
print()
exit()
tim = torchvision.utils.make_grid(torch.cat(images), dim=0)
show_lab_image(tim.unsqueeze(0))
plt.show()
# fig, ax = plt.subplots(figsize=(1, 4))
# rgb_imgs = lab_to_rgb(*split_lab(x))
# plt.imshow(rgb_imgs[0])
# plt.imshow(rgb_imgs[0])
# plt.imshow(rgb_imgs[0])
# plt.show()
# exit()
# print("hi")
# reconstruction = lab_to_rgb(x["L"], x["ab"])

dataset = ColorizationDataset(["./data/bars.jpg"]);
orig_rgb = dataset.get_rgb()
orig_grey = dataset.get_grayscale()
plt.imshow(orig_grey, cmap="gray")
plt.show()
exit()
	
t = torch.Tensor([299]).type(torch.int64)
noised_img = forward_diffusion_sample(x, t)[0]
print(noised_img.shape)
noised_img = torch.nn.functional.normalize(noised_img)
print(torch.max(noised_img))
print(torch.min(noised_img))
noised_rgb = lab_to_rgb(*split_lab(noised_img))
height = 1
width = 4
# plt.subplot(height,  width, 1)
plt.imshow(noised_rgb[0])
plt.show()
# plt.subplot(height, width, 2)
# plt.imshow(orig_rgb)
# plt.subplot(height, width, 3)
# plt.imshow(orig_grey, cmap="gray")
# print(orig_grey.shape)
# plt.ion()




rand_cat_gray = torch.Tensor(orig_grey).unsqueeze(0).unsqueeze(0)
gray_noise = torch.rand((1, 2, 256, 256)) * 2 - 1

# rand_cat_gray = torch.cat((x["L"], gray_noise), dim=1)
# rgb_ver = lab_to_rgb(*split_lab(rand_cat_gray))
# print(f"{rand_cat_gray.shape=}")
# rearranged = rearrange(rgb_ver, "b c h w ->(b h) w c")
# print(f"{rearranged.shape=}")
# plt.subplot(height, width, 4)
# plt.imshow(rearrange(x["L"], "b c h w -> (b h) w c"))
# plt.imshow(rearranged.numpy())
# plt.imshow(rgb_ver[0])
# plt.show()
print()
print(x.max())

# visualize()


