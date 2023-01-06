import glob
import torch
import torchvision
from dataset import ColorizationDataset
from torch.utils.data import DataLoader
from unet import Unet, Encoder
from utils import get_device, lab_to_rgb, split_lab
from default_configs import colordiff_config, unet_config, enc_config
from model import ColorDiffusion

from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser()
    checkpoints = glob.glob("./checkpoints/*.ckpt")
    default_ckpt = checkpoints[-1] if checkpoints else None
    parser.add_argument("-i", "--image-path", required=True, dest="img_path")
    parser.add_argument("-T", "--diffusion-steps", default=350, dest="T")
    parser.add_argument("--image-size", default=64, dest="img_size")
    parser.add_argument("--checkpoint", default=default_ckpt, dest="ckpt")
    parser.add_argument("--show", default=True)
    parser.add_argument("--save", default=True)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()
    assert args.ckpt is not None, "No checkpoint provided and ./checkpoints/ folder empty"

    device = get_device()
    colordiff_config["device"]
    colordiff_config["T"] = args.T
    colordiff_config["img_size"] = args.img_size

    dataset = ColorizationDataset([args.img_path], split="val", config=colordiff_config)
    image = dataset[0].unsqueeze(0)

    encoder = Encoder(**enc_config)
    unet = Unet(**unet_config,) 
    # model = ColorDiffusion.load_from_checkpoint(args.ckpt, strict=True, unet=unet, encoder=encoder, train_dl=None, val_dl=None, **colordiff_config)
    model = ColorDiffusion(strict=True, unet=unet, encoder=encoder, train_dl=None, val_dl=None, **colordiff_config)

    colorized = model.sample_plot_image(image, show=args.show, prog=True)
    rgb_img = lab_to_rgb(*split_lab(colorized))
    if args.save:
        if args.save_path is None:
            save_path = args.img_path +"colorized.jpg"
        save_img = torch.tensor(rgb_img[0]).permute(2, 0, 1)
        torchvision.utils.save_image(save_img, save_path)
    # return(rgb_img[0])
    
