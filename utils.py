from omegaconf import OmegaConf
from glob import glob

import numpy as np
from skimage.color import lab2rgb
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
import torch


def lab_to_pil(img):
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    rgb_img = lab_to_rgb(*split_lab_channels(img))
    pil_img = Image.fromarray(np.uint8(rgb_img[0] * 255))
    return pil_img


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def get_device():
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except:
        device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (device)


def custom_to_pil(x, process=True):
    x = x.detach().cpu()
    if process:
        x = torch.clamp(x, -1., 1.)
        x = (x + 1.)/2.
    x = x.permute(1, 2, 0).numpy()
    if process:
        x = (255*x).astype(np.uint8)
    return x


def show_lab_image(image, stepsize=10, log=True, caption="diff samples"):
    plt.figure(figsize=(20, 9))
    rgb_imgs = lab_to_rgb(*split_lab_channels(image))
    plt.imshow(rgb_imgs[0])
    plt.show()


def init_weights(net, init='norm', gain=2**0.5, leakyslope=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net


def init_model(model, device, init):
    model = init_weights(model, init)
    return model


def right_pad_dims_to(x: torch.tensor, t: torch.tensor) -> torch.tensor:
    """
    Pads `t` with empty dimensions to the number of dimensions `x` has. If `t` does not have fewer dimensions than `x`
        it is returned without change.
    """
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def split_lab_channels(image):
    assert isinstance(image, torch.Tensor)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    return torch.split(image, [1, 2], dim=1)


def cat_lab(L, ab):
    return (torch.cat((L, ab), dim=1))


def lab_to_rgb(L, ab):
    """
    Converts a batch of torch tensors from Lab to RGB
    """
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def l_to_rgb(L):
    """Converts a single channel greyscale image to RGB"""
    if len(L.shape) == 3:
        L = L.unsqueeze(0)
    L = (L + 1.) * 50.
    print(L.min(), L.max())
    return L.repeat(3, dim=1)


def load_default_configs():
    configs = ["./configs/default/encoder_config.yaml", "./configs/default/unet_config.yaml", "./configs/default/colordiff_config.yaml"]
    return [OmegaConf.load(path) for path in configs]
