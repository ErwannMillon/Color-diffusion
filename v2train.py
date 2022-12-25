import wandb
import torch
import pytorch_lightning as pl
from dataset import make_dataloaders, make_dataloaders_celeba
from PLModel import PLColorDiff
from model import ColorDiffusion
from unet import SimpleUnet
from utils import get_device
from pytorch_lightning.loggers import WandbLogger
from icecream import ic
import default_configs
# from stable_diffusion.model.unet import UNetModel
from denoising import Unet, Encoder
from pytorch_lightning.callbacks import ModelCheckpoint

colordiff_config = dict(
    device = "gpu",
    pin_memory = True,
    T=350,
    # lr=6e-4,
    lr = 3e-4,
    loss_fn = "l2",
    batch_size=72,
    accumulate_grad_batches=1,
    img_size = 64,
    sample=True,
    should_log=True,
    epochs=200,
    using_cond=True,
    display_every=100,
    dynamic_threshold=False,
    train_autoenc=False,
    enc_loss_coeff = 1.1,
) 
if __name__ == "__main__":
    colordiff_config["device"] = "gpu" 
    train_dl, val_dl = make_dataloaders_celeba("./img_align_celeba", colordiff_config, num_workers=2, limit=20000)
    log = True

    unet_config = dict(
        channels=3,
        dropout=0.3,
        self_condition=False,
        out_dim=2,
        dim=64,
        condition=True,
        dim_mults=[1, 2, 4, 8],
    )
    enc_config = dict(
        channels=1,
        dropout=0.3,
        self_condition=False,
        out_dim=2,
        dim=64,
        dim_mults=[1, 2, 4, 8],
    )


    encoder = Encoder(
        **enc_config
    )
    unet = Unet(
        **unet_config,
    ) 
    model = ColorDiffusion(unet=unet, encoder=encoder, train_dl=train_dl, val_dl=val_dl, **colordiff_config)
    # model = ColorDiffusion.load_from_checkpoint("Color_diffusion_v2/3cma4wob/checkpoints/last.ckpt", unet=unet, encoder=encoder, train_dl=train_dl, val_dl=train_dl, **colordiff_config)
    # model = torch.compile(model)
    log = True
    colordiff_config["sample"] = log
    colordiff_config["should_log"] = log
    ic.disable()
    if log:
        wandb_logger = WandbLogger(project="Color_diffusion_v2")
        # wandb_logger.watch(model)
        wandb_logger.watch(unet)
        wandb_logger.experiment.config.update(colordiff_config)
        wandb_logger.experiment.config.update(unet_config)
    from pytorch_lightning.profiler import AdvancedProfiler
    ckpt_callback = ModelCheckpoint(every_n_train_steps=300, save_top_k=2, save_last=True, monitor="val_loss")
    profiler = AdvancedProfiler(dirpath="./", filename="profilee")

    # ema = EMA(
    # model,
    # beta = 0.9999,              # exponential moving average factor
    # update_after_step = 100,    # only after this number of .update() calls will it start updating
    # update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
    # )

    trainer = pl.Trainer(max_epochs=colordiff_config["epochs"],
                        logger=wandb_logger if log is True else None, 
                        accelerator=colordiff_config["device"],
                        num_sanity_val_steps=1,
                        devices= "auto",
                        log_every_n_steps=3,
                        callbacks=[ckpt_callback],
                        profiler="simple",
                        accumulate_grad_batches=colordiff_config["accumulate_grad_batches"],
                        # auto_lr_find=True,
                        )
    trainer.fit(model, train_dl, val_dl)
