import gradio as gr
from transformers import pipeline
from PLModel import PLColorDiff
from dataset import ColorizationDataset
from utils import lab_to_rgb, split_lab
import torch
import default_configs
from icecream import ic

from unet import SimpleUnet
def get_image(image):
    print(image)
    dataset = ColorizationDataset([image], split="val", config=conf, size=128)
    lab_img = dataset.get_tensor_from_path(image) 
    batch = lab_img.unsqueeze(0)
    # x_l, _ = split_lab(batch)
    # bw = torch.cat((x_l, *[torch.zeros_like(x_l)] * 2), dim=1)
    model.eval()
    img = model.sample_plot_image(batch)
    rgb_img = lab_to_rgb(*split_lab(img))
    # model.test_step(batch)
    return(rgb_img[0])
conf = SimpleUnetConfig = dict (
    # device = get_device(),
    device = "mps",
    pin_memory = torch.cuda.is_available(),
    T=300,
    lr=5e-4,
    batch_size=64,
    img_size = 128,
    sample=False,
    log=False,
    should_log=False,
    sample_fn = None,
    val_every=20,
    epochs=100,
    using_cond=False
)
ckpt_path = "checkpoints/epoch=1-step=706.ckpt"
ckpt = torch.load(ckpt_path, map_location=torch.device("mps"))
unet = SimpleUnet()
model = PLColorDiff(unet, None, None)
ic.disable()
model.load_state_dict(ckpt["state_dict"])

demo = gr.Interface(
    get_image,
    inputs=gr.inputs.Image(label="Upload a black and white face", type="filepath"),
    outputs="image",
    title="Upload a black and white face and get a colorized image!",
)

demo.launch()