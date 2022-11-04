from cgi import test
from icecream import ic
# from main_model import MainModel
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch
from dataset import make_dataloaders
from dataset import ColorizationDataset, make_dataloaders
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sample import sample_plot_image
from utils import get_device, get_loss, log_results, print_distrib, split_lab, update_losses, visualize, show_lab_image
from main_model import forward_diffusion_sample
import torch.nn.functional as F
import wandb
from unet import SimpleCondUnet, SimpleUnet

#HYPERPARAMS
def optimize_diff(optim, model, batch, device,
                    config, step, e, log=True):
    real_L, real_AB = split_lab(batch[:1, ...].to(device))
    t = torch.randint(0, config["T"], (config["batch_size"],), device=device).long()
    # t = torch.Tensor([1]).to(device).long()
    # show_lab_image(noised_images)
    # show_lab_image(reconstructed_img.detach())
    optim.zero_grad()
    loss = get_loss(model, batch, t, device)
    loss.backward()
    optim.step()
    if (log):
        wandb.log({"epoch":e, "step":step, "loss":loss.item()})

def train_model(model, train_dl, epochs, config, 
                save_interval=15, display_every=200, 
                log=True, ckpt=None, sample=True, writer=None):
    device = config["device"]
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.train()
    optim_diff = torch.optim.Adam(model.parameters(), lr=config["lr_unet"])
    for e in range(epochs):
        for step, batch in tqdm(enumerate(train_dl)):
            real_L, real_AB = split_lab(batch[:1, ...].to(device))
            diff_loss = optimize_diff(optim_diff, model, batch, 
                                        device, config, step, e)

            losses = dict(
                diff_loss = diff_loss.item(),
                )
        if e % save_interval == 0:
            print(f"epoch: {e}, loss {losses}")
            # torch.save(model.state_dict(), f"./saved_models/model_{e}_.pt")
            # for name, weight in model.named_parameters():
            #     writer.add_histogram(name,weight, e)
            #     writer.add_histogram(f'{name}.grad',weight.grad, e)
            # add_to_tb(noise_pred, real_noise, e)
            if sample:
                sample_plot_image(real_L, model, device)

if __name__ == "__main__":
    config = dict (
        batch_size = 1,
        img_size = 128,
        lr_unet = 1e-3,
        device = get_device(),
        T = 300
    )
    writer = SummaryWriter('runs/colordiff')
    wandb.init(project="DiffColor", config=config)
    dataset = ColorizationDataset(["./data/bars.jpg"] * config["batch_size"], config=config);
    train_dl = DataLoader(dataset, batch_size=config["batch_size"])
    device = get_device()
    diff_gen = SimpleUnet().to(device)
    print(f"using device {device}")
    # ckpt = "./saved_models/he_leaky_64.pt"
    ckpt = None
    ic.disable()

    train_model(diff_gen, train_dl, 150,
                ckpt=ckpt, log=True, sample=True,
                save_interval=10, writer=writer, config=config)
############
# def get_loss(model, x_0, t):
#     x_noisy, noise = forward_diffusion_sample(x_0, t, device)
#     noise_pred = model(x_noisy, t)
#     return F.l1_loss(noise, noise_pred)

# from torch.optim import Adam

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
# optimizer = Adam(model.parameters(), lr=0.001)
# epochs = 100 # Try more!
# T = 300
# BATCH_SIZE = 1

# for epoch in range(epochs):
#     for step, batch in enumerate(train_dl):
#       optimizer.zero_grad()

#       t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
#       loss = get_loss(model, batch[0], t)
#       loss.backward()
#       optimizer.step()

#       if epoch % 5 == 0 and step == 0:
#         print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
#         sample_plot_image()