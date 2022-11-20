from dataset import ColorizationDataset, make_dataloaders
from train import config
from torchvision import transforms
import torchvision
import torch
import glob
from matplotlib import pyplot as plt
import random
from utils import show_lab_image
from PIL import Image,ImageChops
from dataset import is_greyscale
from skimage.color import rgb2lab, lab2rgb
import skimage
import numpy as np

config["batch_size"] = 48
def test(path):
    img = Image.open(path).convert("RGB")
    while (is_greyscale(img) is True):
        print("popping")
    # img = self.transforms(img)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
    ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
    return (torch.cat((L, ab), dim=0))
import csv

def make_csv(name, list):
    with open(name, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(list)
def filter_dataset_greyscale():
    train_paths = glob.glob("./fairface" + "/train/*.jpg")
    train_dataset = ColorizationDataset(train_paths, split="train", size=config["img_size"], limit=None)
    print(f"train_dataset.paths = {len(train_dataset.paths)}")
    val_paths = glob.glob("./fairface" + "/val/*.jpg")
    print(len(val_paths))
    val_dataset = ColorizationDataset(val_paths, split="val", size=config["img_size"], limit=None)
    print(len(val_dataset.paths))
    make_csv("val_filtered.csv", val_dataset.paths)
if __name__ == "__main__":
    # print(train_dataset.paths)
    train_paths = glob.glob("./fairface" + "/train/*.jpg")
    print(len(train_paths))
    filter_dataset_greyscale()
    exit()
    # make_csv("train_filtered.csv", train_dataset.paths)
    train_dl, _ = make_dataloaders("./fairface", config=config, limit=200)
    # images = [train_dataset for i in random.sample(range(8000), 100)]
    for images in train_dl:
        # img_stack = torch.stack(list(image))
        grid = torchvision.utils.make_grid(images, dim=0)
        show_lab_image(grid.unsqueeze(0), log=False)
        plt.show()             
    # y = train_dataset.__getitem__([1, 3])
    # x = test("./fairface/train/166.jpg")
    # show_lab_image(x.unsqueeze(0), log=False)
    plt.show()


    
    