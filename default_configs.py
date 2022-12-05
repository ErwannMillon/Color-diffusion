import torch
from utils import get_device
from pytorch_lightning.profiler import SimpleProfiler

SimpleUnetConfig = dict (
    device = get_device(),
    pin_memory = torch.cuda.is_available(),
    T=300,
    lr=5e-4,
    batch_size=64,
    img_size = 64,
    sample=True,
    log=True,
    val_every=20,
    epochs=100,
    using_cond=False
)

StableDiffUnetConfig = dict(
	in_channels=3,
	out_channels=2,
	channels=128,
	attention_levels=[0, 1, 2],
	n_res_blocks=2,
	channel_multipliers=[2, 2, 4, 4],
	n_heads=2,
	tf_layers=1,
	d_cond=512
)

ColorDiffConfig = dict(
    device = get_device(),
    pin_memory = torch.cuda.is_available(),
    T=300,
    lr=5e-4,
    batch_size=8,
    img_size = 64,
    sample=True,
    should_log=False,
    val_every=20,
    epochs=100,
    using_cond=True
) 