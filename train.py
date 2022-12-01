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

def optimize_diff(optim_diff, optim_enc, encoder, diff_gen, batch, device,
                    config, step, e, log=True):
    # real_L, real_AB = split_lab(batch[:1, ...].to(device))
    t = torch.randint(0, config["T"], (batch.shape[0],), device=device).long()
    optim_diff.zero_grad()
    optim_enc.zero_grad()
    diff_loss = get_loss(diff_gen,  batch, t, device)
    diff_loss.backward()
    optim_diff.step()
    optim_enc.step()
    return loss;

def validation_update(step, losses, model, val_dl, config, sample=True, log=True):
    losses["val_loss"] = validation_step(model, val_dl, device, config, sample=True, log=log)
    if log:
        wandb.log(losses)
    return (losses)

def train_model(diff_gen, encoder, train_dl, val_dl, epochs, config, 
                save_interval=15, display_every=200, log=True, ckpt=None, sample=True, writer=None):
    device = config["device"]
    test = ColorizationDataset(["./data/croppedme.jpg"], split="val")
    me = next(iter(test))
    me_xl, _ = split_lab(me.unsqueeze(0))
    if ckpt:
        diff_gen.load_state_dict(torch.load(ckpt, map_location=device))
    diff_gen.train()
    optim_diff = torch.optim.Adam(diff_gen.parameters(), lr=config["lr_unet"])
    optim_enc = torch.optim.Adam(encoder.parameters(), lr=config["lr_enc"])
    for e in range(epochs):
        for step, batch in tqdm(enumerate(train_dl)):
            diff_loss = optimize_diff(optim_diff, optim_enc, encoder, diff_gen, batch, 
                                        device, config, step, e, log=log)
            losses = dict(diff_loss=diff_loss.item(), step = step, epoch=e)
            if step % 20 == 0:
                losses = validation_update(step, losses, diff_gen, val_dl, config, sample=False, log=log)
            if display_every is not None and step % display_every == 0:
                losses = validation_update(step, losses, diff_gen, val_dl, config, sample=True, log=log)
                print(f"epoch: {e}, loss {losses}")
        if e % save_interval == 0:
            losses = validation_update(step, losses, diff_gen, val_dl, config, sample=True, log=log)
            print(f"epoch: {e}, loss {losses}")
            torch.save(diff_gen.state_dict(), f"./saved_models/model_{e}_.pt")


if __name__ == "__main__":
    config = dict (
        batch_size = 32,
        img_size = 64,
        lr_unet = 1e-3,
        device = get_device(),
        pin_memory = torch.cuda.is_available(),
        T = 300
    )
    writer = SummaryWriter('runs/colordiff')
    log = False
    if log: 
        wandb.init(project="DiffColor", config=config)
    # dataset = ColorizationDataset(["./data/bars.jpg"] * config["batch_size"], config=config)
    # train_dl = DataLoader(dataset, batch_size=config["batch_size"])
    train_dl, val_dl = make_dataloaders("./preprocessed_fairface", config)
    device = get_device()
    diff_gen = UNetModel(   in_channels=3,
                            out_channels=2,
                            channels=256,
                            attention_levels=[0, 1, 2],
                            n_res_blocks=2,
                            channel_multipliers=[1, 2, 4, 4],
                            n_heads=2,
                            tf_layers=1,
                            d_cond=512)
    cond_encoder = Encoder( in_channels=1,
                            channels=64,
                            channel_multipliers=[1, 2, 2, 2],
                            n_resnet_blocks=2,
                            z_channels=256 
                            )
    
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

#         print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
#       if epoch % 5 == 0 and step == 0:
#         sample_plot_image()