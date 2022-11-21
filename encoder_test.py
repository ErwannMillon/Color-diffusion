from cgi import test
import glob
from icecream import ic
# from main_model import MainModel
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch
from dataset import make_dataloaders
from dataset import ColorizationDataset, make_dataloaders
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sample import sample_plot_image
from utils import get_device, get_loss, log_results, print_distrib, split_lab, update_losses, visualize, show_lab_image
from main_model import forward_diffusion_sample
import torch.nn.functional as F
import wandb
from unet import SimpleCondUnet, SimpleUnet
from validation import get_val_loss, validation_step
from stable_diffusion.model.unet import UNetModel
import stable_diffusion
from cond_encoder import Encoder
#HYPERPARAMS

config = dict (
    batch_size = 1,
    img_size = 64,
    lr_unet = 1e-3,
    device = get_device(),
    pin_memory = torch.cuda.is_available(),
    T = 300
)

if __name__ == "__main__":
    train_dl, val_dl = make_dataloaders("./preprocessed_fairface", config)
    device = get_device()
    x = next(iter(train_dl)).to(device)
    x_l, _ = split_lab(x)
    diff_gen = UNetModel(   in_channels=3,
                            out_channels=2,
                            channels=256,
                            attention_levels=[0, 1, 2],
                            n_res_blocks=2,
                            channel_multipliers=[1, 2, 4, 4],
                            n_heads=2,
                            tf_layers=1,
                            d_cond=1024)
    cond_encoder = Encoder( in_channels=1,
                            channels=64,
                            channel_multipliers=[1, 2, 2, 2],
                            n_resnet_blocks=2,
                            z_channels=512 
                            )

    diff_gen.to(device)
    cond_encoder.to(device)
    t = torch.randint(0, config["T"], (x.shape[0],), device=device).long()
    cond = cond_encoder(x_l) 
    pred = diff_gen(x, t, cond)
    print(f"using device {device}")
    # ckpt = "./saved_models/he_leaky_64.pt"
    ckpt = None
    ic.disable()
    # train_model(diff_gen, train_dl, val_dl, 1,
    #             ckpt=ckpt, log=log, sample=True, display_every=1,
    #             save_interval=10, writer=writer, config=config)
############
# def get_loss(model, x_0, t):
#     x_noisy, noise = forward_diffusion_sample(x_0, t, device)
#     noise_pred = model(x_noisy, t)
#     return F.l1_loss(noise, noise_pred)

# from torch.optim import Adam

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
# optimizer = Adam(model.parameters(), lr=0.001)
# epochs = 100 # Try more!
# T = 300
# BATCH_SIZE = 1

# for epoch in range(epochs):
#     for step, batch in enumerate(train_dl):
#       optimizer.zero_grad()

#       t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
#       loss = get_loss(model, batch[0], t)
#       loss.backward()
#       optimizer.step()

#       if epoch % 5 == 0 and step == 0:
#         print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
#         sample_plot_image()