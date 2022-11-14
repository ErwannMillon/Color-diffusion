from main_model import MainModel
import torchvision
import torch
from dataset import make_dataloaders
from dataset import ColorizationDataset, make_dataloaders
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from utils import log_results, split_lab, update_losses, visualize, show_lab_image, cat_lab
from main_model import forward_diffusion_sample, linear_beta_schedule
import torch.nn.functional as F
from matplotlib import pyplot as plt
import wandb

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def sample_timestep(x, t, model, T=300):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas = linear_beta_schedule(timesteps=T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    betas_t = get_index_from_list(betas, t, x.shape)
    x_l, x_ab = split_lab(x)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    # model.setup_input(x)
    pred = model(x, t)
    beta_times_pred = betas_t * pred
    model_mean = sqrt_recip_alphas_t * (
        x_ab - beta_times_pred / sqrt_one_minus_alphas_cumprod_t
    )
    # model_mean = sqrt_recip_alphas_t * (
    #     x_ab - betas_t * pred  / sqrt_one_minus_alphas_cumprod_t
    # )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    #TODO Experiment with returning this always
    if t == 0:
        return cat_lab(x_l, model_mean)
    else:
        noise = torch.randn_like(x_ab)
        ab_t_pred = model_mean + torch.sqrt(posterior_variance_t) * noise 
        return cat_lab(x_l, ab_t_pred)

# def sample_plot_image(x_l, model, device, T=300):
def sample_plot_image(val_dl, model, device, T=300):
    model.eval()
    # print("hadsf")
    # # Sample noise
    x = next(iter(val_dl)) 
    x_l, _ = split_lab(x)
    img_size = x_l.shape[-1]
    # print(f"device = {device}")
    x_l = x_l.to(device)
    # print(f"x_l.device = {x_l.device}")
    x_ab = torch.randn((1, 2, img_size, img_size), device=device)
    # print(f"x_ab.device = {x_ab.device}")
    img = torch.cat((x_l, x_ab), dim=1)
    num_images = 10
    stepsize = int(T/num_images)
    images = []
    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model)
        if i % stepsize == 0:
            # print(torch.max(img[:, :1, ...]))
            # print(torch.max(img[:, 1:, ...]))
            # print(torch.min(img[:, :1, ...]))
            # print(torch.min(img[:, :1, ...]))
            img = torch.nn.functional.normalize(img)
            # img = torch.clamp(img, -1, 1) 
            images += img.unsqueeze(0)
            # show_lab_image(img.detach().cpu())
            # show_tensor_image(img.detach().cpu())
    grid = torchvision.utils.make_grid(torch.cat(images), dim=0)
    show_lab_image(grid.unsqueeze(0))
    plt.show()     


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    model = MainModel().to(device)
    ckpt = "./saved_models/he_leaky_64.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    dataset = ColorizationDataset(["./data/bars.jpg"]);
    dataloader = DataLoader(dataset, batch_size=1)
    # x = torch.randn((1, 1, 256, 256))
    x = next(iter(dataloader))
    model.eval()
    sample_plot_image(x[:1, :1,...], model, device)
