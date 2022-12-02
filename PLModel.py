import torch
import pytorch_lightning as pl
from diffusion import forward_diffusion_sample
from sample import sample_plot_image
from utils import split_lab

class PLColorDiff(pl.LightningModule):
    def __init__(self,
                unet, 
                T=300,
                lr=5e-4,
                batch_size=64,
                img_size = 64,
                sample=True,
                log=True,
                sample_fn = sample_plot_image,
                using_cond=False,
                **kwargs):
        super().__init__()
        self.unet = unet.to(self.device)
        self.lr = lr
        self.using_cond = using_cond
        self.sample = sample
        self.T = T
        self.loss = torch.nn.functional.l1_loss
        self.sample_image = sample_fn
    def forward(self, *args):
        return self.unet(args)
    def training_step(self, batch, batch_idx):
        x_0 = batch
        x_l, _ = split_lab(batch)
        t = torch.randint(0, self.T, (batch.shape[0],))
        x_noisy, noise = forward_diffusion_sample(x_0, t, T=self.T)
        if self.using_cond:
            noise_pred = self.unet(x_noisy, t, x_l)
        else:
            noise_pred = self.unet(x_noisy, t)
        loss = self.loss(noise_pred, noise) 
        self.log("train loss", loss)
        return {"loss": loss}
    def validation_step(self, batch, batch_idx):
        val_loss = self.training_step(batch, batch_idx)
        if self.sample:
            x_l, _ = split_lab(batch)
            self.sample_image(x_l, self.unet, self.T, self.log)
        self.log("val loss", val_loss)
        return val_loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.unet.parameters(), lr=self.lr)