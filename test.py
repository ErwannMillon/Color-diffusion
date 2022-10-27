import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from einops import rearrange
from skimage.color import rgb2lab, lab2rgb


import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from main_model import forward_diffusion_sample
from dataset import ColorizationDataset, make_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_colab = None
from utils import lab_to_rgb, visualize

dataset = ColorizationDataset(["./data/test.jpg"]);
dataloader = DataLoader(dataset, batch_size=1)
x = next(iter(dataloader))
reconstruction = lab_to_rgb(x["L"], x["ab"])
orig_rgb = dataset.get_rgb()
orig_grey = dataset.get_grayscale()
# plt.imshow(reconstruction[0])
# plt.ion()
# plt.show()
def split_lab(image):
	assert isinstance(image, torch.Tensor)
	if isinstance(image, torch.Tensor):
		l = image[:, :1,]
		ab = image[:, 1:,]	
	return (l, ab)

	
t = torch.Tensor([299]).type(torch.int64)
noised_img = forward_diffusion_sample(x, t)[0]
print(noised_img.shape)
noised_rgb = lab_to_rgb(*split_lab(noised_img))
height = 1
width = 4
# plt.subplot(height,  width, 1)
# plt.imshow(noised_rgb[0])
# plt.subplot(height, width, 2)
# plt.imshow(orig_rgb)
# plt.subplot(height, width, 3)
# plt.imshow(orig_grey, cmap="gray")
# print(orig_grey.shape)
# plt.ion()



# plt.show()

rand_cat_gray = torch.Tensor(orig_grey).unsqueeze(0).unsqueeze(0)
gray_noise = torch.rand((1, 2, 256, 256)) * 2 - 1

rand_cat_gray = torch.cat((x["L"], gray_noise), dim=1)
rgb_ver = lab_to_rgb(*split_lab(rand_cat_gray))
print(f"{rand_cat_gray.shape=}")
# rearranged = rearrange(rgb_ver, "b c h w ->(b h) w c")
# print(f"{rearranged.shape=}")
# plt.subplot(height, width, 4)
# plt.imshow(rearrange(x["L"], "b c h w -> (b h) w c"))
# plt.imshow(rearranged.numpy())
plt.imshow(rgb_ver[0])
plt.show()


# visualize()

