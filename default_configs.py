import torch
from utils import get_device
from pytorch_lightning.profiler import SimpleProfiler


encoder_conf = dict(
    in_channels=1,
    channels=128,
    channel_multipliers=[1, 1, 2, 3],
    n_resnet_blocks=2,
    z_channels=256
)

celeba_encoder_conf = dict(
  in_channels=1,
  channels=128,
  channel_multipliers=[1, 1, 2, 3],
  n_resnet_blocks=2,
  z_channels=256
)

celeba_unet_config = dict(
    in_channels=3,
    out_channels=2,
    channels=128,
    attention_levels=[1, 2],
    n_res_blocks=2,
    channel_multipliers=[1, 2, 2, 3],
    n_heads=2,
    tf_layers=1,
    d_cond=256
)

celeba_colordiff_config = dict(
    device = "gpu",
    pin_memory = True,
    T=350,
    lr=4e-5,
    batch_size= 8,
    img_size = 64,
    sample=True,
    should_log=True,
    epochs=4,
    using_cond=True,
    display_every=200,
    dynamic_threshold=False,
    train_autoenc=False,
) 

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
	attention_levels=[1, 2],
	n_res_blocks=2,
	# channel_multipliers=[1, 2, 2, 3],
	channel_multipliers=[1, 1, 2],
	n_heads=1,
	tf_layers=1,
	d_cond=256
)

ColorDiffConfig = dict(
    device = get_device(),
    pin_memory = torch.cuda.is_available(),
    T=300,
    lr=5e-4,
    batch_size=1,
    img_size = 64,
    sample=False,
    should_log=False,
    val_every=20,
    epochs=100,
    using_cond=True,
    enc_loss_coeff=0.7,

) 