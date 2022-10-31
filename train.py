from cgi import test
from main_model import MainModel
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch
from dataset import make_dataloaders
from dataset import ColorizationDataset, make_dataloaders
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sample import sample_plot_image
from utils import get_device, log_results, split_lab, update_losses, visualize, show_lab_image
from main_model import forward_diffusion_sample
import torch.nn.functional as F
import wandb

#HYPERPARAMS
def train_model(model, train_dl, epochs, save_interval=15, 
                display_every=200, T=300, batch_size=16,
                log=True, device="cpu", ckpt=None, sample=True):
    # data = next(iter(train_dl)) # getting a batch for visualizing the model output after fixed intrvals
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.unet.train()
    for e in range(epochs):
        for step, batch in tqdm(enumerate(train_dl)):
            # loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
                                                   # log the losses of the complete network
            
            model.setup_input(batch.to(device)) 
            # print(f"{batch.shape=}")
            # print(f"batch.shape = {batch.shape}")
            real_L, real_AB = split_lab(batch[:1, ...].to(device))
            t = torch.randint(0, T, (batch_size,), device=device).long()
            # t = torch.Tensor([1]).to(device).long()
            noised_images, real_noise = forward_diffusion_sample(batch, t, device=device)
            # show_lab_image(noised_images)
            noise_pred, reconstructed_img = model(batch.to(device), t)
            # show_lab_image(reconstructed_img.detach())
            loss = model.optimize(noise_pred, real_noise)
            if (log):
                wandb.log({"epoch":e, "step":step, "loss":loss.item()})
        # if save_interval is not None and e % save_interval == 0:
        if e % save_interval == 0:
            print(f"epoch: {e}, loss {loss.item()}")
            torch.save(model.state_dict(), f"./saved_models/model_{e}_.pt")
            if sample:
                sample_plot_image(real_L, model, device)

def make_graph():
    model = MainModel().to(device)
    t = torch.Tensor([1]).to(device).long()
    test_batch = next(iter(train_dl))
    summary(model, input_data=(test_batch, t), depth=5)
    writer.add_graph(model, (test_batch, t))
    writer.close()

if __name__ == "__main__":
    BATCH_SIZE = 1
    writer = SummaryWriter('runs/colordiff')
    # wandb.init(project="DiffColor", config={"batch_size": BATCH_SIZE, "T": 300})
    dataset = ColorizationDataset(["./data/bars.jpg"] * BATCH_SIZE);
    train_dl = DataLoader(dataset, batch_size=BATCH_SIZE)

    device = get_device()
    model = MainModel().to(device)
    print(f"using device {device}")
    # ckpt = "./saved_models/test.pt"
    ckpt = None
    # for name, param in model.named_parameters():
        # print(name)
        # print(param)
    train_model(model, train_dl, 150, batch_size=BATCH_SIZE, \
                device=device, ckpt=ckpt, log=False, sample=True,\
                save_interval=10)
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