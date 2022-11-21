import stable_diffusion.model.unet
import train
from cgi import test
import glob
from icecream import ic
# from main_model import MainModel
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
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
encoder_defaults = dict(
	in_channels=1,
	channels=64,
	channel_multipliers=[1, 2, 2, 2],
	n_resnet_blocks=1,
	z_channels=256 

)
diffusion_defaults = dict(
	in_channels=3,
	out_channels=2,
	channels=256,
	attention_levels=[1, 2],
	n_res_blocks=2,
	channel_multipliers=[1, 2, 2, 4],
	n_heads=2,
	tf_layers=1,
	d_cond=512
)


class CondColorDiff(nn.Module):
	def __init__(self,
				config, 
				encoder_config=encoder_defaults, 
				diffusion_config=diffusion_defaults) -> None:
		super().__init__() 
		self.device = config["device"]
		self.encoder = Encoder(**encoder_config).to(self.device)
		# self.encoder = torch.nn.DataParallel(self.encoder)
		self.diff_gen = UNetModel(**diffusion_config).to(self.device)
		# self.diff_gen = torch.nn.DataParallel(self.diff_gen)
		self.enc_optim = torch.optim.Adam(self.encoder.parameters(), lr=config["lr_enc"])
		self.diff_optim = torch.optim.Adam(self.diff_gen.parameters(), lr=config["lr_unet"])
	def forward(self, x_t, t, cond_img):
		cond_img = cond_img.to(self.device)
		cond_emb = self.encoder(cond_img)
		noise_pred = self.diff_gen(x_t, t, cond_emb)
		return noise_pred
		# return torch.nn.functional.l1_loss(noise, noise_pred)