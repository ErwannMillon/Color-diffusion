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

if __name__ == "__main__":
    encoder_conf = dict(
        in_channels=1,
        channels=128,
        channel_multipliers=[1, 1, 2, 3],
        n_resnet_blocks=2,
        z_channels=256
    )
    colordiff_config = default_configs.ColorDiffConfig
    colordiff_config["device"] = "mps"
    train_dl, val_dl = make_dataloaders("./fairface", colordiff_config, pickle=False, use_csv=True, num_workers=4)
    log = False
    if log:
        wandb_logger = WandbLogger(project="autoencpretrain")
        wandb_logger.experiment.config.update(encoder_conf)
    autoenc = GreyscaleAutoEnc(encoder_conf,
                                val_dl,
                                display_every=20,
                                should_log=False)
    trainer = pl.Trainer(max_epochs=colordiff_config["epochs"],
                    logger=wandb_logger if log is True else None, 
                    accelerator=colordiff_config["device"],
                    devices=1,
                    val_check_interval=colordiff_config["val_every"])
    trainer.fit(autoenc, train_dl, val_dl)
