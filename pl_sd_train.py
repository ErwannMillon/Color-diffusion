import wandb
import torch
import pytorch_lightning as pl
from cond_encoder import Decoder, Encoder
from dataset import make_dataloaders
from PLModel import PLColorDiff
from unet import SimpleUnet
from utils import get_device
from pytorch_lightning.loggers import WandbLogger
from icecream import ic
import default_configs
from stable_diffusion.model.unet import UNetModel
from autoencoder import GreyscaleAutoEnc

if __name__ == "__main__":
    unet_config = default_configs.StableDiffUnetConfig
    enc_config = default_configs.encoder_conf
    colordiff_config = default_configs.ColorDiffConfig
    # colordiff_config["device"] = "cuda" if torch.cuda.is_available() else "mps" 
    colordiff_config["device"] = "mps" 
    # colordiff_config["device"] = "gpu" 
    # ic.disable()
    # train_dl, val_dl = make_dataloaders("./fairface_preprocessed/preprocessed_fairface", colordiff_config, pickle=True, use_csv=False, num_workers=4)
    train_dl, val_dl = make_dataloaders("./fairface", colordiff_config, pickle=False, use_csv=True, num_workers=4)
    # exit()
    autoenc = GreyscaleAutoEnc(enc_config, val_dl)
    unet = UNetModel(**unet_config)
    model = PLColorDiff(unet, train_dl, val_dl, autoenc, **colordiff_config)
    log = False
    colordiff_config["should_log"] = log
    if log:
        # wandb.login()
        wandb_logger = WandbLogger(project="colordifflocal")
        wandb_logger.watch(model)
        wandb_logger.experiment.config.update(colordiff_config)
        wandb_logger.experiment.config.update(unet_config)
    trainer = pl.Trainer(max_epochs=colordiff_config["epochs"],
                        logger=wandb_logger if log is True else None, 
                        accelerator=colordiff_config["device"],
                        devices=1,
                        val_check_interval=colordiff_config["val_every"])
    trainer.fit(model, train_dl, val_dl)
