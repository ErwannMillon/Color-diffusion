import wandb
import torch
import pytorch_lightning as pl
from dataset import make_dataloaders
from PLModel import PLColorDiff
from sample import sample_plot_image
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
    ic.disable()
    train_dl, val_dl = make_dataloaders("./preprocessed_fairface", colordiff_config)
    unet = UNetModel(**unet_config)
    model = PLColorDiff(unet, train_dl, val_dl, **colordiff_config)
    log = False
    colordiff_config["log"] = log
    if log:
        wandb.login()
        wandb.init(project="sd_colordiff", config=colordiff_config)
        wandb.log(unet_config)
    trainer = pl.Trainer(max_epochs=colordiff_config["epochs"],
                        # logger=wandb_logger, 
                        accelerator=colordiff_config["device"],
                        devices=1,
                        val_check_interval=colordiff_config["val_every"])
    trainer.fit(model, train_dl, val_dl)