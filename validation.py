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

def get_val_loss(model, val_dl, device, config, log=True):
    val_batch = next(iter(val_dl)).detach()
    real_L, real_AB = split_lab(val_batch[:1, ...].to(device))
    t = torch.randint(0, config["T"], (config["batch_size"],), device=device).long()
    return get_loss(model, val_batch, t, device)

def validation_step(model, val_dl, device, config, sample=True, log=True):
    model.eval()
    with torch.no_grad():
        if sample:
            sample_plot_image(val_dl, model, device, log=log)
        val_loss = get_val_loss(model, val_dl, device, config)
        model.train()
        # losses["val_loss"] = val_loss
        return val_loss.detach().item()
    if log:
        wandb.log({"val_loss": val_loss})