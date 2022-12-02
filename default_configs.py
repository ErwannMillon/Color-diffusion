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