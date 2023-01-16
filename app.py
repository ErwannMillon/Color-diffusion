import functools
import gradio as gr
from dataset import ColorizationDataset
from utils import get_device, lab_to_rgb, load_default_configs, split_lab_channels
from icecream import ic
import torch
from model import ColorDiffusion
from denoising import Unet, Encoder
from PIL import Image
import numpy as np


def get_image(model, dataset, image):
    print(image)
    lab_img = dataset.get_lab_from_path(image)
    batch = lab_img.unsqueeze(0).to(device)
    print(batch.shape)
    model.eval()
    with torch.inference_mode():
        img = model.sample_plot_image(batch, show=False, prog=True,
                                      use_ema=False, log=False)
        return img[0]


if __name__ == "__main__":
    enc_config, unet_config, colordiff_config = load_default_configs()
    ckpt = "/home/ubuntu/Color-diffusion/checkpoints/last.ckpt"

    dataset = ColorizationDataset([""], split="val", config=colordiff_config)
    encoder = Encoder(**enc_config)
    unet = Unet(**unet_config)
    model = ColorDiffusion.load_from_checkpoint(ckpt,
                                                strict=True,
                                                unet=unet,
                                                encoder=encoder,
                                                train_dl=None, val_dl=None,
                                                **colordiff_config)

    device = get_device()
    model.to(device)

    infer = functools.partial(get_image, model, dataset)
    with gr.Blocks() as demo:
        with gr.Row():
            image = gr.Image(type="filepath", label="Upload a black and white face")
            out = gr.Image(label="Colorized image")
        image.change(infer, inputs=[image], outputs=[out])
    demo.launch(debug=True)