import wandb
import torch
import pytorch_lightning as pl
from cond_encoder import Encoder
from dataset import make_dataloaders
from PLModel import PLColorDiff
from unet import SimpleUnet
from utils import get_device
from pytorch_lightning.loggers import WandbLogger
from icecream import ic
import default_configs
from stable_diffusion.model.unet import UNetModel

if __name__ == "__main__":
    unet_config = default_configs.StableDiffUnetConfig
    colordiff_config = default_configs.ColorDiffConfig
    colordiff_config["device"] = "mps"
    # ic.disable()
    train_dl, val_dl = make_dataloaders("./preprocessed_fairface",  colordiff_config, num_workers=4)
    unet = UNetModel(**unet_config)
    cond_encoder = Encoder( in_channels=1,
                            channels=64,
                            channel_multipliers=[1, 2, 2, 2],
                            n_resnet_blocks=2,
                            z_channels=512 
                            )
    model = PLColorDiff(unet, train_dl, val_dl, encoder=cond_encoder, **colordiff_config)
    log = True
    colordiff_config["should_log"] = log
    if log:
        # wandb.login()
        wandb_logger = WandbLogger(project="colordifflocal")
        wandb_logger.watch(model)
        wandb_logger.experiment.config.update(colordiff_config)
        wandb_logger.experiment.config.update(unet_config)
    trainer = pl.Trainer(max_epochs=colordiff_config["epochs"],
                        logger=wandb_logger, 
                        accelerator=colordiff_config["device"],
                        devices=1,
                        val_check_interval=colordiff_config["val_every"])
    trainer.fit(model, train_dl, val_dl)
