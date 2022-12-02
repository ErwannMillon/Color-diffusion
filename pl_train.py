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

if __name__ == "__main__":
    wandb.login()
    config = dict (
        device = get_device(),
        pin_memory = torch.cuda.is_available(),
        T=300,
        lr=5e-4,
        batch_size=64,
        img_size = 64,
        sample=True,
        log=True,
        sample_fn = sample_plot_image,
        val_every=20,
        epochs=100,
        using_cond=False
    )
    config["device"] = "cpu"
    ic.disable()
    train_dl, val_dl = make_dataloaders("./preprocessed_fairface", config)
    unet = SimpleUnet()
    model = PLColorDiff(unet, **config)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=config["epochs"],
                        logger=wandb_logger, 
                        accelerator=config["device"],
                        val_check_interval=config["val_every"])
    trainer.fit(model, train_dl, val_dl)
