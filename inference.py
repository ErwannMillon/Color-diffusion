import argparse
import torchvision
from PLModel import PLColorDiff
from dataset import ColorizationDataset
from utils import lab_to_rgb, split_lab
import default_configs
from stable_diffusion.model.unet import UNetModel
import torch
from icecream import ic
from PIL import Image
import PIL

from unet import SimpleUnet
def get_image(model, img_path, show=True, save=False, save_path=None):
    # print(img_path)
    dataset = ColorizationDataset([img_path], split="val", config=conf, size=128)
    lab_img = dataset.get_tensor_from_path(img_path) 
    batch = lab_img.unsqueeze(0)
    model.eval()
    img = model.sample_plot_image(batch, show=show, prog=True)
    rgb_img = lab_to_rgb(*split_lab(img))
    if save:
        if save_path is None:
            save_path = img_path+"colorized.jpg"
        save_img = torch.tensor(rgb_img[0]).permute(2, 0, 1)
        torchvision.utils.save_image(save_img, save_path)
    return(rgb_img[0])
conf = SimpleUnetConfig = dict (
    # device = get_device(),
    device = "mps",
    pin_memory = torch.cuda.is_available(),
    T=300,
    lr=5e-4,
    batch_size=64,
    img_size = 64,
    sample=False,
    log=False,
    should_log=False,
    sample_fn = None,
    val_every=20,
    epochs=100,
    using_cond=False
)
if __name__ == "__main__":
    from default_configs import StableDiffUnetConfig
    unet_config = default_configs.StableDiffUnetConfig
    colordiff_config = default_configs.ColorDiffConfig

    colordiff_config["device"] = "mps"
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./data/croppedme.jpg")
    parser.add_argument("--show", default=True)
    parser.add_argument("--save", default=True)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()
    ckpt_path = "saved_models/kaggle.ckpt"
    ckpt = torch.load(ckpt_path, map_location=torch.device("mps"))
    # unet = SimpleUnet()
    unet = UNetModel(**unet_config)
    model = PLColorDiff(unet, None, None, **colordiff_config)
    ic.disable()
    model.load_state_dict(ckpt["state_dict"])
    get_image(model, args.path, show=args.show, save=args.save)

