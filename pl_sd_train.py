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
# from ema_pytorch import EMA


encoder_conf = dict(
    in_channels=1,
    channels=64,
    channel_multipliers=[1, 2, 2],
    n_resnet_blocks=2,
    z_channels=128
)

small_unet = dict(
    in_channels=2,
    out_channels=2,
    channels=64,
    attention_levels=[0, 1, 2],
    n_res_blocks=2,
    channel_multipliers=[1, 2, 3],
    # channe
    n_heads=1,
    tf_layers=1,
    d_cond=128
)

unet_config = dict(
    in_channels=2,
    out_channels=2,
    channels=128,
    attention_levels=[1, 2],
    n_res_blocks=2,
    channel_multipliers=[1, 2, 3, 3],
    # channe
    n_heads=2,
    tf_layers=1,
    d_cond=128
)

colordiff_config = dict(
    device = "gpu",
    pin_memory = True,
    T=350,
    # lr=6e-4,
    lr = 1e-5,
    batch_size=3,
    img_size = 128,
    sample=True,
    should_log=True,
    epochs=5,
    using_cond=True,
    display_every=20,
    dynamic_threshold=False,
    train_autoenc=False,
    enc_loss_coeff = 1.1,
) 

if __name__ == "__main__":
    # unet_config = default_configs.StableDiffUnetConfig
    # colordiff_config = default_configs.ColorDiffConfig
    # colordiff_config["device"] = "cuda" if torch.cuda.is_available() else "mps" 
    # colordiff_config["device"] = "mps" 
    colordiff_config["device"] = "gpu" 
    # ic.disable()
    # train_dl, val_dl = make_dataloaders("./fairface_preprocessed/preprocessed_fairface", colordiff_config, pickle=True, use_csv=False, num_workers=4)
    train_dl, val_dl = make_dataloaders_celeba("./img_align_celeba", colordiff_config, num_workers=2, limit=150)
    log = True
    # exit()
    autoenc = GreyscaleAutoEnc.load_from_checkpoint("autoencpretrain/2gyi15mo/checkpoints/epoch=4-step=1600.ckpt",  
                                            encoder_config=encoder_conf,
                                            val_dl=val_dl,
                                            display_every=50,
                                            should_log=False)
    # autoenc = GreyscaleAutoEnc(encoder_config=encoder_conf, val_dl=val_dl, display_every=50, should_log=False)
    unet = UNetModel(**unet_config)
    model = PLColorDiff(unet, train_dl, val_dl, autoenc, **colordiff_config)
    # model = PLColorDiff.load_from_checkpoint("/home/ec2-user/Color-diffusion/Color_diffusion_dec/azure0.ckpt", unet=unet, train_dl=train_dl, val_dl=val_dl, autoencoder=autoenc, **colordiff_config)
    log = True
    colordiff_config["sample"] = log
    colordiff_config["should_log"] = log
    ic.disable()
    if log:
        wandb_logger = WandbLogger(project="Color_diffusion_dec")
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
                        log_every_n_steps=1,
                        callbacks=[ckpt_callback],
                        profiler="simple",
                        accumulate_grad_batches=5,
                        # auto_lr_find=True,
                        )
    trainer.fit(model, train_dl, val_dl)
