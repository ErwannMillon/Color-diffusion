from dataset import make_dataloaders
import torch
from utils import lab_to_rgb, split_lab
from PIL import Image
from diffusion import GaussianDiffusion
import imageio
import glob
from omegaconf import OmegaConf
import os

def clear_img_dir(img_dir):
    if not os.path.exists("img_history"):
        os.mkdir("img_history")
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for filename in glob.glob(img_dir+"/*"):
        os.remove(filename)

def create_gif(total_duration, extend_frames, folder="./img_history", gif_name="face_edit.gif"):
    images = []
    paths = glob.glob(folder + "/*")
    frame_duration = total_duration / len(paths)
    print(len(paths), "frame dur", frame_duration)
    durations = [frame_duration] * len(paths)
    if extend_frames:
        durations [0] = 1.5
        durations [-1] = 1.5
    for file_name in os.listdir(folder):
        if file_name.endswith('.png'):
            file_path = os.path.join(folder, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(gif_name, images, duration=durations)
    return gif_name

if __name__ == "__main__":
    colordiff_config = OmegaConf.load("configs/default/colordiff_config.yaml")
    colordiff_config["batch_size"] = 1
    _, val_dl = make_dataloaders("./img_align_celeba", colordiff_config)
    batch = next(iter(val_dl))
    diffusion = GaussianDiffusion(300)
    
    for i in range(300):
        img, _ = diffusion.forward_diff(batch, torch.tensor([i]))
        pil_img = Image.fromarray(lab_to_rgb(*split_lab(img)))
        pil_img.save(f"./visualization/forward_diff/{i:04d}.png")

    # create_gif()