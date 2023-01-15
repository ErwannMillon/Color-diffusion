import torch
from torch import nn
from torch import optim
from utils import init_weights, lab_to_pil
import torch.nn.functional as F
from dynamic_threshold import dynamic_threshold
from utils import cat_lab, split_lab_channels
from pytorch_lightning import LightningModule

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    vals = vals.to(t)
    out = vals.gather(-1, t.long())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class GaussianDiffusion(LightningModule):
    def __init__(self, T, dynamic_threshold=False) -> None:
        super().__init__()
        self.betas = linear_beta_schedule(timesteps=T).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.dynamic_threshold=dynamic_threshold
    def forward_diff(self, x_0, t, T=300):
        """ 
        Takes an image and a timestep as input and noises the color channels to timestep t
        """
        l, ab = split_lab_channels(x_0)
        noise = torch.randn_like(ab)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, ab.shape).to(x_0)
        # print(f"sqrt_alphas_cumprod_t = {sqrt_alphas_cumprod_t}")
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, ab.shape
        ).to(x_0)
        # print(f"sqrt_one_minus_alphas_cumprod_t = {sqrt_one_minus_alphas_cumprod_t}")
        # mean + variance
        ab_noised = sqrt_alphas_cumprod_t * ab \
        + sqrt_one_minus_alphas_cumprod_t * noise

        noised_img = torch.cat((l, ab_noised), dim=1)
        # lab_to_pil(noised_img).save("noised_img.png")
        # print(f"noise = {noise}")

        return(noised_img, noise)

    @torch.no_grad()
    def sample_timestep(self, model, encoder, x, t, cond=None, T=300, ema=None):
        x_l, x_ab = split_lab_channels(x)
        #gets the mean- and variance-derived variables for timestep t 
        betas_t = get_index_from_list(self.betas.to(x), t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)

        # Call model (current image - noise prediction)
        greyscale_emb = encoder(x_l)
        if ema is not None:
            with ema.average_parameters():
                pred = model(x, t, greyscale_emb)
        else:
            pred = model(x, t, greyscale_emb)
        beta_times_pred = betas_t * pred
        model_mean = sqrt_recip_alphas_t * (
            x_ab - beta_times_pred / sqrt_one_minus_alphas_cumprod_t
        )
        if t == 0:
            if self.dynamic_threshold:
                model_mean = dynamic_threshold(model_mean)
            return cat_lab(x_l, model_mean)
        else:
            noise = torch.randn_like(x_ab)
            ab_t_pred = model_mean + torch.sqrt(posterior_variance_t) * noise 
            if self.dynamic_threshold:
                ab_t_pred = dynamic_threshold(ab_t_pred)
            return cat_lab(x_l, ab_t_pred)
if __name__ == "__main__":
    d = GaussianDiffusion(T=300)