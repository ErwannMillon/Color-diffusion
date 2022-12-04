import torch
from tqdm import tqdm
import pytorch_lightning as pl
from diffusion import forward_diffusion_sample, get_index_from_list, linear_beta_schedule
from sample import dynamic_threshold, sample_plot_image
from utils import cat_lab, show_lab_image, split_lab
import torch.nn.functional as F
import torchvision
import wandb
from matplotlib import pyplot as plt
class PLColorDiff(pl.LightningModule):
    def __init__(self,
                unet, 
                train_dl,
                val_dl,
                T=300,
                lr=5e-4,
                enc_lr=3e-4,
                batch_size=64,
                img_size = 64,
                sample=True,
                should_log=True,
                using_cond=False,
                display_every=None,
                **kwargs):
        super().__init__()
        self.unet = unet.to(self.device)
        self.T = T
        self.lr = lr
        self.enc_lr = enc_lr
        self.using_cond = using_cond
        self.sample = sample
        self.loss = torch.nn.functional.l1_loss
        self.should_log=should_log
        if sample is True and display_every is None:
            display_every = 1000
        self.display_every = display_every
        self.val_dl = val_dl
        self.train_dl = train_dl
        # self.save_hyperparameters()
    def forward(self, *args):
        return self.unet(args)
    def training_step(self, batch, batch_idx):
        x_0 = batch
        x_l, _ = split_lab(batch)
        t = torch.randint(0, self.T, (batch.shape[0],)).to(x_0)
        x_noisy, noise = forward_diffusion_sample(x_0, t, T=self.T)
        if self.using_cond:
            noise_pred = self.unet(x_noisy, t, x_l)
        else:
            noise_pred = self.unet(x_noisy, t)
        if self.sample and batch_idx % self.display_every == 0:
            self.test_step(batch)
        loss = self.loss(noise_pred, noise) 
        if self.should_log:
            # wandb.log({"tloss": loss})
            self.log("train loss", loss)
        return {"loss": loss}
    def validation_step(self, batch, batch_idx):
        val_loss = self.training_step(batch, batch_idx)
        if self.should_log:
            # wandb.log({"v_loss": val_loss})
            self.log("val loss", val_loss)
        if self.sample and batch_idx % self.display_every == 0:
            self.sample_plot_image(batch)
        return val_loss
    def test_step(self, batch, *args, **kwargs):
        x = next(iter(self.val_dl)).to(batch)
        self.sample_plot_image(x)
    def configure_optimizers(self):
        unet_optim = torch.optim.Adam(self.unet.parameters(), lr=self.lr)
        return unet_optim


    @torch.no_grad()
    def sample_timestep(self, x, t, cond=None, T=300):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        betas = linear_beta_schedule(timesteps=T).to(x)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        betas_t = get_index_from_list(betas, t, x.shape)
        x_l, x_ab = split_lab(x)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
        
        # Call model (current image - noise prediction)
        if self.using_cond and cond:
            pred = self.unet(x, t, cond)
        else:
            pred = self.unet(x, t)
        beta_times_pred = betas_t * pred
        model_mean = sqrt_recip_alphas_t * (
            x_ab - beta_times_pred / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
        
        if t == 0:
            model_mean = dynamic_threshold(model_mean)
            return cat_lab(x_l, model_mean)
        else:
            noise = torch.randn_like(x_ab)
            ab_t_pred = model_mean + torch.sqrt(posterior_variance_t) * noise 
            ab_t_pred = dynamic_threshold(ab_t_pred)
            return cat_lab(x_l, ab_t_pred)
    @torch.no_grad()
    def sample_plot_image(self, x_0, show=True, prog=False):
        images = []
        if x_0.shape[1] == 3:
            x_l, _ = split_lab(x_0)
            x_l.to(x_0)
            images.append(x_0[:1])
        else:
            x_l = x_0
        x_l = x_l[:1]
        img_size = x_l.shape[-1]
        bw = torch.cat((x_l, *[torch.zeros_like(x_l)] * 2), dim=1)
        images += bw.unsqueeze(0)
        if len(x_l.shape) == 3:
            x_l = x_l.unsqueeze(0)
        x_ab = torch.randn((x_l.shape[0], 2, img_size, img_size)).to(x_l)
        img = torch.cat((x_l, x_ab), dim=1)
        num_images = 10
        stepsize = self.T//num_images
        counter = tqdm(range(0, self.T)[::-1]) if prog else range(0, self.T)[::-1]
        for i in counter:
            t = torch.full((1,), i, dtype=torch.long).to(img)
            img = self.sample_timestep(img, t, self.T)
            if i % stepsize == 0:
                images += img.unsqueeze(0)
        grid = torchvision.utils.make_grid(torch.cat(images), dim=0).to(x_l)
        if show is False:
            return images[-1]
        show_lab_image(grid.unsqueeze(0), log=self.should_log)
        plt.show()     
        return images[-1]
