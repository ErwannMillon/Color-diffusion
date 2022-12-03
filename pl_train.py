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

if __name__ == "__main__":
    wandb.login()
    config = default_configs.SimpleUnetConfig
    config["device"] = "mps"
    ic.disable()
    train_dl, val_dl = make_dataloaders("./preprocessed_fairface", config)
    unet = SimpleUnet()
    model = PLColorDiff(unet, **config)
    # wandb_logger = WandbLogger()
    wandb.init(project="localdiffcolor")
    trainer = pl.Trainer(max_epochs=config["epochs"],
                        # logger=wandb_logger, 
                        accelerator=config["device"],
                        devices=1,
                        val_check_interval=config["val_every"])
    trainer.fit(model, train_dl, val_dl)
