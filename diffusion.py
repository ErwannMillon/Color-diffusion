import torch
from torch import nn
from torch import optim
from models import init_weights
import torch.nn.functional as F
from utils import split_lab

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def forward_diffusion_sample(x_0, t, T=300):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    betas = linear_beta_schedule(timesteps=T)

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    if isinstance(x_0, torch.Tensor):
        l, ab = split_lab(x_0)
    else:
        l = x_0["L"]
        ab = x_0["ab"]

    noise = torch.randn_like(ab)
    # noise = torch.nn.functional.normalize(noise)

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, ab.shape).to(x_0)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, ab.shape
    ).to(x_0)
    # mean + variance
    ab_noised = sqrt_alphas_cumprod_t * ab \
    + sqrt_one_minus_alphas_cumprod_t * noise

    noised_img = torch.cat((l, ab_noised), dim=1)
    return(noised_img, noise)