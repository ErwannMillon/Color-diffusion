from utils import right_pad_dims_to, split_lab, show_lab_image, cat_lab
import torch
from einops import rearrange

def dynamic_threshold(img, percentile=0.8):
    s = torch.quantile(
        rearrange(img, 'b ... -> b (...)').abs(),
        percentile,
        dim=-1
    )
    # If threshold is less than 1, simply clamp values to [-1., 1.]
    s.clamp_(min=1.)
    s = right_pad_dims_to(img, s)
    # Clamp to +/- s and divide by s to bring values back to range [-1., 1.]
    img = img.clamp(-s, s) / s
    return img

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    from train import config
    # train_dl, val_dl = make_dataloaders("./fairface", config)
    # ckpt = "./saved_models/he_leaky_64.pt"
    # model.load_state_dict(torch.load(ckpt, map_location=device))
    # model = SimpleUnet()
    # model.eval()
    # ic.disable()
    # x = next(iter(val_dl))[:1,]
    # x_l, _ = split_lab(x) # print(f"device = {device}")
    # sample_plot_image(None, model, device, x_l=x_l, log=False)
