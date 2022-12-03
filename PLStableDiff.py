import torch
from stable_diffusion.sampler import ddim
from PLModel import PLColorDiff

class PLStableDiff(PLColorDiff):
    def __init__(self,
                unet, 
                encoder,
                train_dl,
                val_dl,
                T=1000,
                lr=5e-4,
                batch_size=64,
                img_size = 64,
                sample=True,
                should_log=True,
                using_cond=False,
                display_every=None,
                **kwargs):
        super().__init__(unet, train_dl, val_dl, T, lr, batch_size, img_size, sample, should_log, using_cond, display_every, **kwargs)
        self.unet = unet
        self.encoder = encoder
