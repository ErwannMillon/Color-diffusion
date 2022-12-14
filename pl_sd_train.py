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

encoder_conf = dict(
    in_channels=1,
    channels=64,
    channel_multipliers=[1, 1, 2, 3],
    n_resnet_blocks=2,
    z_channels=256
)

unet_config = dict(
    in_channels=3,
    out_channels=2,
    channels=128,
    attention_levels=[1, 2],
    n_res_blocks=2,
    channel_multipliers=[1, 2, 3, 3],
    # channe
    n_heads=2,
    tf_layers=1,
    d_cond=256
)

colordiff_config = dict(
    device = "gpu",
    pin_memory = True,
    T=350,
    lr=6e-4,
    # lr = 0.00017378008287493763,
    batch_size=6,
    img_size = 128,
    sample=True,
    should_log=True,
    epochs=2,
    using_cond=True,
    display_every=200,
    dynamic_threshold=False,
    train_autoenc=False,
    enc_loss_coeff = 1.5,
) 

if __name__ == "__main__":
    # unet_config = default_configs.StableDiffUnetConfig
    # colordiff_config = default_configs.ColorDiffConfig
    # colordiff_config["device"] = "cuda" if torch.cuda.is_available() else "mps" 
    # colordiff_config["device"] = "mps" 
    colordiff_config["device"] = "gpu" 
    # ic.disable()
    # train_dl, val_dl = make_dataloaders("./fairface_preprocessed/preprocessed_fairface", colordiff_config, pickle=True, use_csv=False, num_workers=4)
    train_dl, val_dl = make_dataloaders_celeba("./celeba/img_align_celeba", colordiff_config, num_workers=16, limit=120000)
    log = True
    # exit()
    colordiff_config["should_log"] = True
    colordiff_config["sample"] = True
    autoenc = GreyscaleAutoEnc.load_from_checkpoint("800ae.ckpt",  
                                            encoder_config=encoder_conf,
                                            val_dl=val_dl,
                                            display_every=50,
                                            should_log=False)
    unet = UNetModel(**unet_config)
    model = PLColorDiff(unet, train_dl, val_dl, autoenc, **colordiff_config)
    log = True
    colordiff_config["should_log"] = log
    ic.disable()
    if log:
        wandb_logger = WandbLogger(project="colordifflocal")
        # wandb_logger.watch(model)
        wandb_logger.experiment.config.update(colordiff_config)
        wandb_logger.experiment.config.update(unet_config)
    from pytorch_lightning.profiler import AdvancedProfiler
    ckpt_callback = ModelCheckpoint(every_n_train_steps=300)
    profiler = AdvancedProfiler(dirpath="./", filename="profilee")
    trainer = pl.Trainer(max_epochs=colordiff_config["epochs"],
                        logger=wandb_logger if log is True else None, 
                        accelerator=colordiff_config["device"],
                        num_sanity_val_steps=1,
                        devices= "auto",
                        log_every_n_steps=1,
                        callbacks=[ckpt_callback],
                        profiler="simple",
                        # accumulate_grad_batches=8,
                        # auto_lr_find=True,
                        )
                        
    print("""**************
   HARDCODED LR WARNING 
    
   *********** 
    """)
    trainer.fit(model, train_dl, val_dl, ckpt_path="colordifflocal/36ajfifc/805mainmodel.ckpt")
