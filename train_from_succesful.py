import wandb
import torch
import pytorch_lightning as pl
from cond_encoder import Decoder, Encoder
from dataset import make_dataloaders, make_dataloaders_celeba
from PLModel import PLColorDiff
from unet import SimpleUnet
from utils import get_device
from pytorch_lightning.loggers import WandbLogger
from icecream import ic
import default_configs
from stable_diffusion.model.unet import UNetModel
from autoencoder import GreyscaleAutoEnc
from pytorch_lightning.callbacks import ModelCheckpoint

unet_config = dict(
    in_channels=3,
    out_channels=2,
    channels=128,
    attention_levels=[1, 2],
    n_res_blocks=2,
    channel_multipliers=[1, 2, 2, 3],
    n_heads=2,
    tf_layers=1,
    d_cond=256
)

colordiff_config = dict(
    device = "gpu",
    pin_memory = True,
    T=350,
    lr=6e-5,
    batch_size= 40,
    img_size = 64,
    sample=True,
    should_log=True,
    epochs=9,
    using_cond=True,
    display_every=70,
    dynamic_threshold=False,
    train_autoenc=False,
) 

encoder_conf = dict(
  in_channels=1,
  channels=128,
  channel_multipliers=[1, 1, 2, 3],
  n_resnet_blocks=2,
  z_channels=256
)

if __name__ == "__main__":
    # unet_config = default_configs.StableDiffUnetConfig
    # colordiff_config = default_configs.ColorDiffConfig
    # colordiff_config["device"] = "cuda" if torch.cuda.is_available() else "mps" 
    # colordiff_config["device"] = "mps" 
    # colordiff_config["device"] = "gpu" 
    # # ic.disable()
    # # train_dl, val_dl = make_dataloaders("./fairface_preprocessed/preprocessed_fairface", colordiff_config, pickle=True, use_csv=False, num_workers=4)
    train_dl, val_dl = make_dataloaders_celeba("./celeba/img_align_celeba", colordiff_config, num_workers=8, limit=120000)
    # log = True
    # # exit()
    # colordiff_config["should_log"] = True
    # colordiff_config["sample"] = True
    # autoenc = GreyscaleAutoEnc.load_from_checkpoint("800ae.ckpt",  
    #                                         encoder_config=encoder_conf,
    #                                         val_dl=val_dl,
    #                                         display_every=50,
    #                                         should_log=False)
    # unet = UNetModel(**unet_config)
    # model = PLColorDiff(unet, train_dl, val_dl, autoenc, **colordiff_config)
    log = True
    colordiff_config["should_log"] = log
    ic.disable()
    ckpt_callback = ModelCheckpoint(every_n_train_steps=110)
    unet = UNetModel(**unet_config)
    autoenc = GreyscaleAutoEnc.load_from_checkpoint("ae.ckpt", encoder_config=encoder_conf,
                                            val_dl=val_dl,
                                            display_every=70,
                                            should_log=log)
    model = PLColorDiff(unet, train_dl, val_dl, autoenc, **colordiff_config)
    for param in model.autoenc.parameters():
        param.requires_grad = False
    if log:
        wandb_logger = WandbLogger(project="colordifflocal")
        # wandb_logger.watch(model)
        wandb_logger.watch(model)
        wandb_logger.watch(model.unet)
        # wandb_logger.watch(model.autoenc)
        wandb_logger.experiment.config.update(colordiff_config)
        wandb_logger.experiment.config.update(unet_config)
    from pytorch_lightning.profiler import AdvancedProfiler
    profiler = AdvancedProfiler(dirpath="./", filename="profilee")
    trainer = pl.Trainer(max_epochs=colordiff_config["epochs"],
                        logger=wandb_logger if log is True else None, 
                        accelerator=colordiff_config["device"],
                        num_sanity_val_steps=1,
                        default_root_dir="./saved_models",
                        devices= "auto",
                        log_every_n_steps=4,
                        callbacks=[ckpt_callback],
                        profiler="simple",
                        # accumulate_grad_batches=8,
                        # auto_lr_find=True,
                        )
    trainer.fit(model, train_dl, val_dl, ckpt_path="colordifflocal/103zg8m0/checkpoints/epoch=3-step=1980.ckpt")