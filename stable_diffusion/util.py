"""
---
title: Utility functions for stable diffusion
summary: >
 Utility functions for stable diffusion
---

# Utility functions for [stable diffusion](index.html)
"""

import os
import random
from pathlib import Path

import PIL
import numpy as np
import torch
from PIL import Image

from labml import monit
from labml.logger import inspect
from latent_diffusion import LatentDiffusion
from model.autoencoder import Encoder, Decoder, Autoencoder
from model.clip_embedder import CLIPTextEmbedder
from model.unet import UNetModel


def set_seed(seed: int):
    """
    ### Set random seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(ckpt: str = None) -> LatentDiffusion:
    unet_model = UNetModel(in_channels=3,
                            out_channels=2,
                            channels=256,
                            attention_levels=[0, 1, 2],
                            n_res_blocks=2,
                            channel_multipliers=[1, 2, 4, 4],
                            n_heads=2,
                            tf_layers=1,
                            d_cond=512)

    # Initialize the Latent Diffusion model
    model = LatentDiffusion(linear_start=0.00085,
                            linear_end=0.0120,
                            n_steps=350,
                            unet_model=unet_model)
                            # latent_scaling_factor=0.18215,
                            # autoencoder=autoencoder,
                            # clip_embedder=clip_text_embedder,

    # Load the checkpoint
    if ckpt:
        checkpoint = torch.load(ckpt, map_location="cpu")
        missing_keys, extra_keys = model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_img(path: str):
    """
    ### Load an image

    This loads an image from a file and returns a PyTorch tensor.

    :param path: is the path of the image
    """
    # Open Image
    image = Image.open(path).convert("RGB")
    # Get image size
    w, h = image.size
    # Resize to a multiple of 32
    w = w - w % 32
    h = h - h % 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    # Convert to numpy and map to `[-1, 1]` for `[0, 255]`
    image = np.array(image).astype(np.float32) * (2. / 255.0) - 1
    # Transpose to shape `[batch_size, channels, height, width]`
    image = image[None].transpose(0, 3, 1, 2)
    # Convert to torch
    return torch.from_numpy(image)


def save_images(images: torch.Tensor, dest_path: str, prefix: str = '', img_format: str = 'jpeg'):
    """
    ### Save a images

    :param images: is the tensor with images of shape `[batch_size, channels, height, width]`
    :param dest_path: is the folder to save images in
    :param prefix: is the prefix to add to file names
    :param img_format: is the image format
    """

    # Create the destination folder
    os.makedirs(dest_path, exist_ok=True)

    # Map images to `[0, 1]` space and clip
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # Transpose to `[batch_size, height, width, channels]` and convert to numpy
    images = images.cpu().permute(0, 2, 3, 1).numpy()

    # Save images
    for i, img in enumerate(images):
        img = Image.fromarray((255. * img).astype(np.uint8))
        img.save(os.path.join(dest_path, f"{prefix}{i:05}.{img_format}"), format=img_format)
