import torch
from PIL import Image
from tqdm import tqdm
from diffusion import GaussianDiffusion
from dynamic_threshold import dynamic_threshold
from utils import cat_lab, freeze_module, lab_to_rgb, show_lab_image, split_lab, init_weights
import torch.nn.functional as F
import torchvision
import wandb
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from copy import deepcopy
# from torch_ema import ExponentialMovingAverage

class ColorDiffusion(pl.LightningModule):
    def __init__(self,
                unet, 
                train_dl,
                val_dl,
                encoder,
                loss_fn="l2",
                T=300,
                lr=1e-4,
                batch_size=12,
                img_size = 64,
                sample=True,
                should_log=True,
                upscaler=None,
                using_cond=False,
                display_every=None,
                dynamic_threshold=False,
                use_ema=True,
                **kwargs):
        super().__init__()
        self.unet = unet.to(self.device)
        self.T = T
        self.lr = lr
        self.using_cond = using_cond
        self.sample = sample
        self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=0.9999)
        self.encoder = encoder
        self.ema.to("cuda")
        self.diffusion = GaussianDiffusion(T, dynamic_threshold=dynamic_threshold)
        self.loss_fn = torch.nn.functional.l1_loss if loss_fn == "l1" else torch.nn.functional.mse_loss
        self.should_log=should_log
        if sample is True and display_every is None:
            display_every = 1000
        self.display_every = display_every
        self.val_dl = val_dl
        self.train_dl = train_dl
        self.save_hyperparameters(ignore=['unet'])
    def forward(self, x_noised, t, x_l):
        """
        Performs one denoising step on batch of noised inputs
        Unet is conditioned on timestep and features extracted from the greyscale channel
        """
        cond = self.encoder(x_l) # if self.using_cond else None
        noise_pred = self.unet(x_noised, t, greyscale_embs=cond)
        return noise_pred
    def get_batch_pred(self, x_0, x_l):
        """
        Samples a timestep from range [0, T]
        Adds noise to images x_0 to get x_t (x_0 with color channels noised)
        Returns:
        - The model's prediction of the noise, 
        - The real noise applied to the color channels by the forward diffusion process
        """
        t = torch.randint(0, self.T, (x_0.shape[0],)).to(x_0)
        x_noised, noise = self.diffusion.forward_diff(x_0, t, T=self.T)
        return (self(x_noised, t, x_l), noise)
    def get_losses(self, noise_pred, noise, x_l):
        diff_loss = self.loss_fn(noise_pred, noise)
        return {"total loss": diff_loss}
    def training_step(self, x_0, batch_idx):
        x_l, _ = split_lab(x_0)
        noise_pred, noise = self.get_batch_pred(x_0, x_l)
        losses = self.get_losses(noise_pred, noise, x_l)
        self.log_dict(losses, on_step=True)
        if self.sample and batch_idx and batch_idx % self.display_every == 0 and self.global_step > 1:
            self.test_step(x_0)
        return losses["total loss"]
    def validation_step(self, batch, batch_idx):
        x_l, _ = split_lab(batch)
        noise_pred, noise = self.get_batch_pred(batch, x_l)
        losses = self.get_losses(noise_pred, noise, x_l)
        # val_loss = self.training_step(batch, batch_idx)
        if self.should_log:
            self.log("val_loss", losses["total loss"])
        if self.sample and batch_idx and batch_idx % self.display_every == 0:
            self.sample_plot_image(batch)
        return losses["total loss"]
    @torch.inference_mode()
    def test_step(self, batch, *args, **kwargs):
        x = next(iter(self.val_dl)).to(batch)
        self.sample_plot_image(x)
        self.sample_plot_image(x, use_ema=True)
    def configure_optimizers(self):
        learnable_params = list(self.unet.parameters()) + list(self.encoder.parameters())
        global_optim = torch.optim.AdamW(learnable_params, lr=self.lr, weight_decay=28e-3)
        return global_optim
    def log_img(self, image, caption="diff samples", use_ema=False):
        rgb_imgs = lab_to_rgb(*split_lab(image))
        if use_ema:
            self.logger.log_image("EMA samples", [rgb_imgs])
        else:
            self.logger.log_image("samples", [rgb_imgs])
    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update()
    @torch.inference_mode()
    def sample_loop(self, x_l, prog=False, use_ema=False, save_all=False):
        """
        Noises color channels to timestep T, then denoises the color channels to t=0 to get the colorized image
        Returns an array containing the noised image, intermediate images in the denoising process, and the final image
        """
        ema = self.ema if use_ema else None
        images = []
        num_images = 13
        img_size = x_l.shape[-1]
        stepsize = self.T // num_images

        #Initialize image with random noise in color channels
        x_ab = torch.randn((x_l.shape[0], 2, img_size, img_size)).to(x_l)
        img = torch.cat((x_l, x_ab), dim=1)

        counter = range(0, self.T)[::-1]
        if prog: 
            counter = tqdm(counter)
        for i in counter:
            t = torch.full((1,), i, dtype=torch.long).to(img)
            img = self.diffusion.sample_timestep(self.unet, self.encoder, img, t, T=self.T, cond=x_l, ema=ema)
            if i % stepsize == 0:
                images += img.unsqueeze(0)
            #TODO Debug
            if save_all:
                pil_img = Image.fromarray(lab_to_rgb(*split_lab(img)))
                pil_img.save(f"./tmp/frame{i}.png")
        return images

    @torch.inference_mode()
    def sample_plot_image(self, x_0, show=True, prog=False, use_ema=False, log=True):
        """
        Denoises a single image and displays a grid showing the ground truth, intermediate outputs in the denoising process, and the final denoised image
        """
        ground_truth_images = []
        if x_0.shape[1] == 3: #if the image has color channels
            x_l, _ = split_lab(x_0)
            ground_truth_images.append(x_0[:1])
        else:
            x_l = x_0
        x_l = x_l[:1]
        greyscale = torch.cat((x_l, *[torch.zeros_like(x_l)] * 2), dim=1) 
        ground_truth_images += greyscale.unsqueeze(0)
        if len(x_l.shape) == 3:
            x_l = x_l.unsqueeze(0)
        images = ground_truth_images + self.sample_loop(x_l, prog=prog, use_ema=use_ema)
        grid = torchvision.utils.make_grid(torch.cat(images), dim=0).to(x_l)
        if show:
            show_lab_image(grid.unsqueeze(0), log=self.should_log)
            plt.show()     
        if self.should_log and log:
            self.log_img(grid.unsqueeze(0), use_ema=use_ema)
        return lab_to_rgb(*split_lab(images[-1]))
