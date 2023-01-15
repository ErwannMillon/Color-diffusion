import functools
import gradio as gr
from dataset import ColorizationDataset
from utils import lab_to_rgb, split_lab
from icecream import ic
import torch
from model import ColorDiffusion
from super_image import MdsrModel, ImageLoader
from default_configs import unet_config, enc_config, colordiff_config
from denoising import Unet, Encoder
from PIL import Image
import numpy as np

def get_image(model, super_res, image):
    print(image)
    dataset = ColorizationDataset([image], split="val", config=colordiff_config, size=64)
    lab_img = dataset.get_tensor_from_path(image) 
    batch = lab_img.unsqueeze(0)
    print(batch.shape)
    model.eval()
    with torch.inference_mode():
        img = model.sample_plot_image(batch, show=False, prog=True, use_ema=False, log=False)
        rgb_img = lab_to_rgb(*split_lab(img))
        PIL_image = Image.fromarray(np.uint8(rgb_img[0] * 255)).convert('RGB')
        inputs = ImageLoader.load_image(PIL_image)
        upscaled = super_res(inputs)
        upscaled_pil = Image.fromarray(upscaled[0].detach().numpy().transpose(1, 2, 0).astype(np.uint8))
    return(rgb_img[0], upscaled_pil)

encoder = Encoder(**enc_config)
unet = Unet(**unet_config,) 
device = "cuda"
ckpt = "/home/ec2-user/Color-diffusion/Color_diffusion_v2/23l96nt1/checkpoints/last.ckpt"
model = ColorDiffusion.load_from_checkpoint(ckpt, strict=True, unet=unet, encoder=encoder, train_dl=None, val_dl=None, **colordiff_config)
super_res = MdsrModel.from_pretrained('eugenesiow/mdsr', scale=4)      # scale 2, 3 and 4 models available
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