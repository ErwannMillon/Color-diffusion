import functools
import gradio as gr
from denoising import Unet, Encoder
from dataset import ColorizationDataset
from utils import lab_to_rgb, split_lab
from icecream import ic
import wandb
import torch
import pytorch_lightning as pl
from dataset import make_dataloaders, make_dataloaders_celeba
from model import ColorDiffusion
from unet import SimpleUnet
from pytorch_lightning.loggers import WandbLogger
from icecream import ic
from super_image import MdsrModel, ImageLoader
from denoising import Unet, Encoder
from PIL import Image
from unet import SimpleUnet
import numpy as np

def get_image(model, super_res, image):
    print(image)
    dataset = ColorizationDataset([image], split="val", config=colordiff_config, size=64)
    lab_img = dataset.get_tensor_from_path(image) 
    batch = lab_img.unsqueeze(0)
    print(batch.shape)
    model.eval()
    with torch.inference_mode():
        img = model.sample_plot_image(batch, show=False, prog=True, use_ema=True, log=False)
        rgb_img = lab_to_rgb(*split_lab(img))
        PIL_image = Image.fromarray(np.uint8(rgb_img[0] * 255)).convert('RGB')
        inputs = ImageLoader.load_image(PIL_image)
        upscaled = super_res(inputs)
        upscaled_pil = Image.fromarray(upscaled[0].detach().numpy().transpose(1, 2, 0).astype(np.uint8))
    # print(upscaled_pil.shape)
    return(rgb_img[0], upscaled_pil)

colordiff_config = dict(
    device = "gpu",
    pin_memory = True,
    T=29,
    # lr=6e-4,
    lr = 1e-6,
    loss_fn = "l2",
    batch_size=32,
    accumulate_grad_batches=2,
    img_size = 64,
    sample=True,
    should_log=True,
    epochs=14,
    using_cond=True,
    display_every=200,
    dynamic_threshold=False,
    train_autoenc=False,
    enc_loss_coeff = 1.1,
) 
unet_config = dict(
    channels=3,
    dropout=0.3,
    self_condition=False,
    out_dim=2,
    dim=128,
    condition=True,
    dim_mults=[1, 2, 3, 3],
)
enc_config = dict(
    channels=1,
    dropout=0.3,
    self_condition=False,
    out_dim=2,
    dim=128,
    dim_mults=[1, 2, 3, 3],
)
encoder = Encoder(
    **enc_config
)
unet = Unet(
    **unet_config,
) 
device = "cuda"
ckpt = "/home/ec2-user/Color-diffusion/Color_diffusion_v2/23l96nt1/checkpoints/last.ckpt"
super_res = MdsrModel.from_pretrained('eugenesiow/mdsr', scale=4)      # scale 2, 3 and 4 models available
model = ColorDiffusion.load_from_checkpoint(ckpt, strict=True, unet=unet, encoder=encoder, train_dl=None, val_dl=None, **colordiff_config)
ic.disable()
infer = functools.partial(get_image, model, super_res)
demo = gr.Interface(
    infer,
    inputs=[gr.inputs.Image(label="Upload a black and white face", type="filepath")],
    outputs="image",
    title="Upload a black and white face and get a colorized image!",
)
with gr.Blocks() as demo:
    with gr.Row():
        image = gr.Image(type="filepath", label="Upload a black and white face")
        out = gr.Image(label="Colorized image")
        upscaled = gr.Image(label="Upscaled colorized image")
    image.change(infer, inputs=[image], outputs=[out, upscaled])

demo.launch(debug=True)