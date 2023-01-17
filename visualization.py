import glob
import torch
from dataset import ColorizationDataset, make_dataloaders, make_datasets
from torch.utils.data import DataLoader
from denoising import Unet, Encoder
from utils import get_device, lab_to_rgb, load_default_configs, split_lab_channels
from model import ColorDiffusion
from argparse import ArgumentParser

from dataset import make_dataloaders
from matplotlib import pyplot as plt
from PIL import Image
from diffusion import GaussianDiffusion
import imageio
import numpy as np
from omegaconf import OmegaConf
import os


def clear_img_dir(img_dir):
    if not os.path.exists("img_history"):
        os.mkdir("img_history")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for filename in glob.glob(img_dir+"/*"):
        os.remove(filename)

def create_gif_full(folder, total_duration, extend_frames=True, gif_name="face_edit.gif"):
    images = []
    paths = list(sorted(glob.glob(folder + "/*")))[::2] + list(reversed(sorted(glob.glob("./visualization/denoising" + "/*"))))
    # print(paths)
    frame_duration = total_duration / len(paths)
    print(len(paths), "frame dur", frame_duration)
    durations = [frame_duration] * len(paths)
    if extend_frames:
        durations [0] = 1.5
        durations [-1] = 1.5
    for file_path in paths:
        images.append(imageio.imread(file_path))
    imageio.mimsave(gif_name, images, duration=durations)
    return gif_name

def create_gif(folder, total_duration, extend_frames=True, gif_name="face_edit.gif"):
    images = []
    paths = list(sorted(glob.glob(folder + "/*")))
    print(paths)
    frame_duration = total_duration / len(paths)
    print(len(paths), "frame dur", frame_duration)
    durations = [frame_duration] * len(paths)
    if extend_frames:
        durations[0] = 1.5
        durations[-1] = 1.5
    for file_path in paths:
        images.append(imageio.imread(file_path))
    imageio.mimsave(gif_name, images, duration=durations)
    return gif_name


def visualize_forward(batch):
    diffusion = GaussianDiffusion(300)
    for i in range(300):
        t = torch.tensor([i]).float()
        img, _ = diffusion.forward_diff(batch.cuda(), t=t)
        rgb_img = lab_to_rgb(*split_lab_channels(img))
        pil_img = Image.fromarray(np.uint8(rgb_img[0] * 255))
        pil_img.save(f"./visualization/forward_diff/{i:04d}.png")


def visualize_backward(model):
    model.sample_plot_image(batch.to(device), show=args.show,
                            prog=True, save_all=True, log=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    # checkpoints = glob.glob("./checkpoints/last.ckpt")
    # default_ckpt = checkpoints[-1] if checkpoints else None
    default_ckpt = "./checkpoints/last.ckpt"
    parser.add_argument("-i", "--image-path", required=True, dest="img_path")
    parser.add_argument("-T", "--diffusion-steps", default=350, dest="T")
    parser.add_argument("--image-size", default=64, dest="img_size", type=int)
    parser.add_argument("--checkpoint", default=default_ckpt, dest="ckpt")
    parser.add_argument("--show", default=True)
    parser.add_argument("--save", default=True)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()
    assert args.ckpt is not None, "No checkpoint provided and ./checkpoints/ folder empty"

    device = get_device()
    enc_config, unet_config, colordiff_config = load_default_configs()
    colordiff_config["T"] = args.T
    colordiff_config["img_size"] = args.img_size
    colordiff_config["should_log"] = False
    colordiff_config["batch_size"] = 1
    colordiff_config["img_size"] = 64
    _, val_dl = make_dataloaders("./img_align_celeba", colordiff_config)
    dataset, _ = make_datasets("./img_align_celeba", colordiff_config)
    batch = next(iter(val_dl))

    encoder = Encoder(**enc_config)
    unet = Unet(**unet_config,) 
    model = ColorDiffusion.load_from_checkpoint(args.ckpt, strict=True,
                                                unet=unet, encoder=encoder,
                                                train_dl=None, val_dl=None,
                                                **colordiff_config)
    model.to(device)
    for i in range(10):

        batch = dataset.get_lab_grayscale(idx=i+123)
        visualize_forward(batch.to(device))
        visualize_backward(model)
        create_gif_full("./visualization/forward_diff", 5, gif_name=f"./visualization/total_{i}.gif")
        
        
