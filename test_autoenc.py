from cgi import test
import glob
from icecream import ic
# 
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch
from dataset import make_dataloaders
from dataset import ColorizationDataset, make_dataloaders
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from utils import get_device, get_loss, log_results, print_distrib, split_lab, update_losses, visualize, show_lab_image
import torch.nn.functional as F
import wandb
from unet import SimpleCondUnet, SimpleUnet
from validation import get_val_loss, validation_step
from stable_diffusion.model.unet import UNetModel
import stable_diffusion
from autoencoder import GreyscaleAutoEnc
from cond_encoder import Decoder, Encoder
from CondColorDiff import CondColorDiff
import default_configs
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from dataset import make_dataloaders_celeba

if __name__ == "__main__":
    encoder_conf = dict(
        in_channels=1,
        channels=64,
        channel_multipliers=[1, 1, 2, 3],
        n_resnet_blocks=2,
        z_channels=256
    )
    colordiff_config = dict(
        device = "gpu",
        pin_memory = True,
        T=350,
        lr=1e-4,
        batch_size=32,
        img_size = 128,
        sample=True,
        should_log=True,
        epochs=1,
        using_cond=True,
        display_every=200,
        dynamic_threshold=False,
        train_autoenc=True,
        enc_loss_coeff = 1.5,
    ) 
    # colordiff_config = default_configs.ColorDiffConfig
    colordiff_config["device"] = "gpu"
    # colordiff_config["device"] = "mps"
    train_dl, val_dl = make_dataloaders_celeba("./celeba/img_align_celeba", colordiff_config, num_workers=4, limit=30000)
    log = True
    if log:
        wandb_logger = WandbLogger(project="autoencpretrain")
        wandb_logger.experiment.config.update(encoder_conf)
    from pytorch_lightning.callbacks import ModelCheckpoint
    ckpt_callback = ModelCheckpoint(every_n_train_steps=400)
    autoenc = GreyscaleAutoEnc(encoder_conf,
                                val_dl,
                                display_every=20,
                                should_log=True)
    trainer = pl.Trainer(max_epochs=colordiff_config["epochs"],
                    logger=wandb_logger if log is True else None, 
                    accelerator=colordiff_config["device"],
                    devices="auto",
                    callbacks=[ckpt_callback],
                    log_every_n_steps=2
                    )
    trainer.fit(autoenc, train_dl, val_dl, ckpt_path="800ae.ckpt")
