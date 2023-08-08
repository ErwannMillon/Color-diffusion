import glob
import torch
import torchvision
from dataset import ColorizationDataset, make_dataloaders
from denoising import Unet, Encoder
from utils import get_device, lab_to_rgb, load_default_configs, \
                    split_lab_channels
from model import ColorDiffusion
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    checkpoints = glob.glob("./checkpoints/last.ckpt")
    default_ckpt = checkpoints[-1] if checkpoints else None
    # default_ckpt = "./checkpoints/last.ckpt"

    parser.add_argument("-i", "--image-path", required=True, dest="img_path")
    parser.add_argument("-T", "--diffusion-steps", default=350, dest="T")
    parser.add_argument("--image-size", default=64, dest="img_size", type=int)
    parser.add_argument("--checkpoint", default=default_ckpt, dest="ckpt")
    parser.add_argument("--show", default=True)
    parser.add_argument("--save", default=True)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()
    assert args.ckpt is not None, "No checkpoint passed and ./checkpoints/ folder empty"

    device = get_device()
    enc_config, unet_config, colordiff_config = load_default_configs()
    print("loaded default model config")
    colordiff_config["T"] = args.T
    colordiff_config["img_size"] = args.img_size

    dataset = ColorizationDataset([args.img_path],
                                  split="val",
                                  config=colordiff_config)
    image = dataset[0].unsqueeze(0)

    encoder = Encoder(**enc_config)
    unet = Unet(**unet_config)
    model = ColorDiffusion.load_from_checkpoint(args.ckpt,
                                                strict=True,
                                                unet=unet,
                                                encoder=encoder,
                                                train_dl=None,
                                                val_dl=None,
                                                **colordiff_config)
    model.to(device)

    colorized = model.sample_plot_image(image.to(device),
                                        show=args.show,
                                        prog=True)
    rgb_img = lab_to_rgb(*split_lab_channels(colorized))
    if args.save:
        if args.save_path is None:
            save_path = args.img_path + "colorized.jpg"
        save_img = torch.tensor(rgb_img[0]).permute(2, 0, 1)
        torchvision.utils.save_image(save_img, save_path)
