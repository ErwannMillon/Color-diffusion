from cond_encoder import Encoder, Decoder
import wandb
import torchvision
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import nn

from utils import lab_to_rgb, show_lab_image, split_lab

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
        del decoder_config["in_channels"]
        self.lr = lr
        self.display_every = display_every
        self.should_log = should_log
        self.val_dl = val_dl
        self.decoder = Decoder(**decoder_config)
        self.mse = torch.nn.functional.mse_loss
    def forward(self, x):
        x_l, _ = split_lab(x)
        x_l = x_l.to(x)
        emb = self.encoder(x_l)
        rec = self.decoder(emb)
        return rec
    def training_step(self, batch, batch_idx):
        x_l, _ = split_lab(batch)
        rec = self(batch)
        loss = self.mse(rec, x_l)
        if self.should_log:
            self.log("train loss", loss, on_step=True)
        if batch_idx % self.display_every == 0:
            self.test_step(batch)
        return loss
    def test_step(self, batch, *args):
        x = next(iter(self.val_dl)).to(batch)
        #TODO Check if split lab return xl when given xl only
        x = x[:1,]
        x_l, _ = split_lab(x)
        rec = self(x.to(batch))
        if self.should_log:
            loss = self.mse(rec, x_l)
            self.log("val loss", loss, on_step=True)
        plt.figure(figsize=(20, 20))
        ab = torch.zeros((1, 2, x.shape[-1], x.shape[-1])).to(x)
        images = [torch.cat((x_l, ab), dim=1), torch.cat((rec, ab), dim=1)]
        # rgb_ground_truth = lab_to_rgb(x_l, ab)
        # rgb_rec = lab_to_rgb(rec, ab)
        # images=[rgb_ground_truth, rgb_rec]
        grid = torchvision.utils.make_grid(torch.cat(images), dim=0).to(x_l)
        show_lab_image(grid.unsqueeze(0), log=self.should_log)
        plt.show()
    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return loss
    def configure_optimizers(self):
        learnable_params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        global_optim = torch.optim.Adam(learnable_params, lr=self.lr)
        return global_optim