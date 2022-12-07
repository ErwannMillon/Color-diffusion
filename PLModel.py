import torch
from tqdm import tqdm
import pytorch_lightning as pl
from diffusion import GaussianDiffusion
from sample import dynamic_threshold
from utils import cat_lab, freeze_module, lab_to_rgb, show_lab_image, split_lab
import torch.nn.functional as F
import torchvision
import wandb
from matplotlib import pyplot as plt
from cond_encoder import Encoder
from icecream import ic
class PLColorDiff(pl.LightningModule):
    def __init__(self,
                unet, 
                train_dl,
                val_dl,
                autoencoder,
                enc_loss_coeff=.5,
                enc_lr=3e-4,
                T=300,
                lr=5e-4,
                batch_size=64,
                img_size = 64,
                sample=True,
                should_log=True,
                using_cond=False,
                display_every=None,
                train_autoenc=False,
                dynamic_threshold=True,
                **kwargs):
        super().__init__()
        self.unet = unet.to(self.device)
        self.T = T
        self.lr = lr
        self.using_cond = using_cond
        self.sample = sample
        self.diffusion = GaussianDiffusion(T, dynamic_threshold=dynamic_threshold)
        self.l1 = torch.nn.functional.l1_loss
        self.l2 = torch.nn.functional.mse_loss
        self.should_log=should_log
        if sample is True and display_every is None:
            display_every = 1000
        self.display_every = display_every
        self.val_dl = val_dl
        self.train_dl = train_dl
        self.autoenc = autoencoder
        self.autoenc_frozen = False
        self.train_autoenc = train_autoenc
        if train_autoenc is False:
            del self.autoenc.decoder
            freeze_module(self.autoenc)
            self.autoenc_frozen = True
        self.enc_loss_coeff = enc_loss_coeff
        self.save_hyperparameters(ignore=['unet', 'autoencoder'])
        # self.enc_lr = enc_lr
    def forward(self, x_noisy, t, x_l):
        """
        Performs one denoising step on batch of inputs noised with timesteps t, and makes an embedding of the original greyscale channel to condition the unet if self.using_cond is True
        """
        if self.using_cond:
            assert x_l is not None
            cond_emb = self.autoenc.encoder(x_l)
            noise_pred = self.unet(x_noisy, t, cond_emb)
            if self.train_autoenc:
                x_l_rec = self.autoenc.decoder(cond_emb)
            else: x_l_rec = None
        else:
            noise_pred = self.unet(x_noisy, t)
        return noise_pred, x_l_rec
    def get_batch_pred(self, x_0, x_l):
        """
        Gets model's predictions of noise at timestep t for batch x_0, and returns:
        - The model's prediction of the noise, 
        - The x_l channel reconstructed by the autoencoder (if training the autoencoder alongside the Unet)
        - The real noise applied to the image by the forward diffusion process
        """
        t = torch.randint(0, self.T, (x_0.shape[0],)).to(x_0)
        x_noisy, noise = self.diffusion.forward_diff(x_0, t, T=self.T)
        return (*self(x_noisy, t, x_l), noise)
    def get_losses(self, noise_pred, noise, x_l_rec, x_l):
        rec_loss = 0.
        diff_loss = self.l2(noise_pred, noise)
        train_loss = diff_loss
        if self.train_autoenc:
            rec_loss = self.l2(x_l_rec, x_l)
            train_loss += self.enc_loss_coeff * rec_loss
        return {"train loss": train_loss,
                "rec loss": rec_loss,
                "diff loss": diff_loss}
    def training_step(self, x_0, batch_idx):
        x_l, _ = split_lab(x_0)
        noise_pred, x_l_rec, noise = self.get_batch_pred(x_0, x_l)
        losses = self.get_losses(noise_pred, noise, x_l_rec, x_l)
        self.log_dict(losses, on_step=True)
        if self.sample and batch_idx and batch_idx % self.display_every == 0:
            self.test_step(x_0)
        if batch_idx > 200 and self.autoenc_frozen is False:
            print ("freezing autoencoder")
            freeze_module(self.autoenc)
            torch.save(self.autoenc.state_dict(), "./earlystop_ae.pt")
            self.autoenc_frozen = True
            self.train_autoenc = False
        return losses["train loss"]
    def validation_step(self, batch, batch_idx):
        val_loss = self.training_step(batch, batch_idx)
        if self.should_log:
            self.log("val loss", val_loss, on_step=True)
        if self.sample and batch_idx and batch_idx % self.display_every == 0:
            self.sample_plot_image(batch)
        return val_loss
    def test_step(self, batch, *args, **kwargs):
        x = next(iter(self.val_dl)).to(batch)
        self.sample_plot_image(x)
    def configure_optimizers(self):
        learnable_params = list(self.unet.parameters()) + list(self.autoenc.parameters())
        # learnable_params = self.unet.parameters() 
        global_optim = torch.optim.Adam(learnable_params, lr=self.lr)
        return global_optim
    
    @torch.no_grad()
    def sample_loop(self, x_l, prog=False):
        """
        Noises the image to timestep T (color channels are random)
        Then autoregressively denoises the color channels to t=0 to get the colorized image
        Returns an array containing the noised image, intermediate images in the denoising process, and the final image
        """
        images = []
        img_size = x_l.shape[-1]
        num_images = 12
        stepsize = self.T // num_images
        #Initialize image with random noise in color channels
        x_ab = torch.randn((x_l.shape[0], 2, img_size, img_size)).to(x_l)
        img = torch.cat((x_l, x_ab), dim=1)
        counter = range(0, self.T)[::-1]
        if prog: counter = tqdm(counter)
        #Progressively 
        for i in counter:
            t = torch.full((1,), i, dtype=torch.long).to(img)
            img = self.diffusion.sample_timestep(self.unet, img, t, T=self.T, cond=x_l, encoder=self.autoenc.encoder)
            if i % stepsize == 0:
                images += img.unsqueeze(0)
        return images
    @torch.no_grad()
    def sample_plot_image(self, x_0, show=True, prog=False):
        """
        Denoises a single image and displays a grid showing the ground truth, intermediate outputs in the denoising process, and the final denoised image
        """
        ground_truth_images = []
        if x_0.shape[1] == 3: #if the image has color channels
            x_l, _ = split_lab(x_0)
            ground_truth_images.append(x_0[:1]) #add ground truth to image grid
        else:
            x_l = x_0
        x_l = x_l[:1]
        greyscale = torch.cat((x_l, *[torch.zeros_like(x_l)] * 2), dim=1) 
        ground_truth_images += greyscale.unsqueeze(0) #add greyscale version of ground truth (model input before noising) to image grid
        if len(x_l.shape) == 3:
            x_l = x_l.unsqueeze(0)
        images = ground_truth_images + self.sample_loop(x_l, prog=prog)
        grid = torchvision.utils.make_grid(torch.cat(images), dim=0).to(x_l)
        if show:
            show_lab_image(grid.unsqueeze(0), log=self.should_log)
            _ = lab_to_rgb(*split_lab(images[-1])) #debugging (this warns about pixels that are outside of valid LAB color range only for the final output image)
            plt.show()     
        return images[-1]
