from cond_encoder import Encoder, Decoder
import wandb
import torchvision
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import nn

from utils import lab_to_rgb

class GreyscaleAutoEnc(pl.LightningModule):
    def __init__(self, 
                encoder_config,
                val_dl,
                lr=1e-4,
                img_size = 64,
                should_log=True,
                sample=True,
                display_every=200,
                ) -> None:
        super().__init__()
        self.encoder = Encoder(**encoder_config)
        decoder_config = encoder_config
        decoder_config["channel_multipliers"] = list(reversed(encoder_config["channel_multipliers"]))
        decoder_config["out_channels"] = encoder_config["in_channels"] 
        self.lr = lr
        self.should_log = should_log
        self.val_dl = val_dl
        self.decoder = Decoder(**decoder_config)
        self.mse = torch.nn.functional.mse_loss
    def forward(self, x_l):
        emb = self.encoder(x_l)
        rec = self.decoder(emb)
        return rec
    def training_step(self, batch, batch_idx):
        rec = self(batch)
        loss = self.mse(rec, batch)
        if self.should_log:
            self.log("train loss", loss)
        if batch_idx % self.display_every == 0:
            self.test_step(batch)
        return loss
    def test_step(self, batch, *args):
        x_l = next(iter(self.val_dl))
        x_l = x_l[:1,]
        rec = self(x_l)
        plt.figure(20, 20)
        rgb_ground_truth = lab_to_rgb(x_l, torch.zeros_like(x_l))
        rgb_rec = lab_to_rgb(rec, torch.zeros_like(x_l))
        images=[rgb_ground_truth, rgb_rec]
        grid = torchvision.utils.make_grid(torch.cat(images), dim=0)
        if self.should_log:
            image = wandb.Image(grid.unsqueeze(0))
            wandb.log({"examples": images})
        plt.imshow(grid)
    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return loss
    def configure_optimizers(self):
        learnable_params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        global_optim = torch.optim.Adam(learnable_params, lr=self.lr)