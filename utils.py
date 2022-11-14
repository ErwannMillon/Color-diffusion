import time
import wandb
from icecream import ic

import numpy as np
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

import torch

from main_model import forward_diffusion_sample
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    return (device)

def split_lab(image):
	assert isinstance(image, torch.Tensor)
	if isinstance(image, torch.Tensor):
		l = image[:, :1,]
		ab = image[:, 1:,]	
	return (l, ab)

def show_lab_image(image, stepsize=10, log=True):
    # image = torch.nn.functional.normalize(image)
    # image = torch.clamp(image, -1, 1)
    plt.figure(figsize=(20, 9))
    rgb_imgs = lab_to_rgb(*split_lab(image))
    images = wandb.Image(rgb_imgs, caption=f"x_0 to x_300 in steps of {stepsize}")
    if log:
        wandb.log({"examples": images})
    plt.imshow(rgb_imgs[0])
    # for i in range(rgb_imgs.shape[0]):
    #     plt.subplot(1, rgb_imgs.shape[0], i + 1)
    # plt.figure(figsize=(10, 10))
    # plt.ion()
    # plt.show()
    
def init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device, init):
    model = model.to(device)
    model = init_weights(model, init)
    return model
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def get_loss(model, x_0, t, device="cuda"):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return torch.nn.functional.l1_loss(noise, noise_pred)

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def right_pad_dims_to(x: torch.tensor, t: torch.tensor) -> torch.tensor:
    """
    Pads `t` with empty dimensions to the number of dimensions `x` has. If `t` does not have fewer dimensions than `x`
        it is returned without change.
    """
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def split_lab(image):
	assert isinstance(image, torch.Tensor)
	if isinstance(image, torch.Tensor):
		l = image[:, :1,]
		ab = image[:, 1:,]	
	return (l, ab)

def cat_lab(L, ab):
    return (torch.cat((L, ab), dim=1))

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def print_distrib(x, str=None):
    assert isinstance(x, torch.Tensor)
    if str is not None:
        print(str)
    ic(x.max())
    ic(x.min())
    ic(x.mean())
    ic(x.std())
    print()

        
def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")


def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")
