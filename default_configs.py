import torch
from sample import sample_plot_image
from utils import get_device
SimpleUnetConfig = dict (
    device = get_device(),
    pin_memory = torch.cuda.is_available(),
    T=300,
    lr=5e-4,
    batch_size=64,
    img_size = 64,
    sample=True,
    log=True,
    sample_fn = sample_plot_image,
    val_every=20,
    epochs=100,
    using_cond=False
)

StableDiffUnetConfig = dict(
	in_channels=3,
	out_channels=2,
	channels=64,
	attention_levels=[],
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
    batch_size=64,
    img_size = 64,
    sample=True,
    should_log=False,
    sample_fn = sample_plot_image,
    val_every=20,
    epochs=100,
    using_cond=True
) 