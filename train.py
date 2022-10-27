from sqlite3 import register_converter
from main_model import MainModel
import torch
from dataset import make_dataloaders
from dataset import ColorizationDataset, make_dataloaders
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from utils import log_results, update_losses, visualize, show_lab_image
from main_model import forward_diffusion_sample
import torch.nn.functional as F
import wandb

#HYPERPARAMS
BATCH_SIZE = 16
# wandb.init(project="DiffColor", config={"batch_size": BATCH_SIZE, "T": 300})

dataset = ColorizationDataset(["./data/test.jpg"] * 16);
train_dl = DataLoader(dataset, batch_size=BATCH_SIZE)
def train_model(model, train_dl, epochs, save_interval=500, 
                display_every=200, T=300, batch_size=16, device="cpu"):
    # data = next(iter(train_dl)) # getting a batch for visualizing the model output after fixed intrvals
    model.unet.train()
    for e in range(epochs):
        for step, batch in tqdm(enumerate(train_dl)):
            # loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
                                                   # log the losses of the complete network
            
            model.setup_input(batch.to(device)) 
            # print(f"{batch.shape=}")
            # print(f"batch.shape = {batch.shape}")
            t = torch.randint(0, T, (batch_size,), device=device).long()
            noised_images, real_noise = forward_diffusion_sample(batch, t)
            # show_lab_image(noised_images)
            noise_pred, reconstructed_img = model(batch.to(device), t)
            # show_lab_image(reconstructed_img.detach())
            loss = model.optimize(noise_pred, real_noise)
            # wandb.log({"epoch":e, "step":step, "loss":loss.item()})
            # update_losses(model, loss_meter_dict, count=batch['L'].size(0)) # function updating the log objects
            if step % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {step}")
                show_lab_image(reconstructed_img.detach())
            if step % save_interval == 0:
                torch.save(model.state_dict(), f"./saved_models/model_{e}_{step}")
                # log_results(loss_meter_dict) # function to print out the losses
                # visualize(model, batch, save=False) # function displaying the model's outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")
model = MainModel()
train_model(model, train_dl, 100, device=device)
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