import torch
from torch import nn
import math
from icecream import ic

from utils import print_distrib, split_lab


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, scale=1):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(int(2 * scale) * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(scale * in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.LeakyReLU(0.02)
        # self.tanh = nn.Tanh()
        
    def forward(self, x, t, ):
        # First Conv
        # ic()
        # print_distrib(x)
        h = self.conv1(x)
        # ic()
        # print_distrib(h)
        h = self.relu(h)
        # ic()
        # print_distrib(h)
        h = self.bnorm1(h)
        # ic()
        # print_distrib(h)
        # h = self.bnorm1(self.tanh(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # ic()
        # print_distrib(h)
        # Second Conv
        h = self.conv2(h)
        # ic()
        # print_distrib(h)
        h = self.relu(h)
        # ic()
        # print_distrib(h)
        h = self.bnorm2(h)
        # ic()
        # print_distrib(h)
        # h = self.bnorm2(self.tanh(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class CondBlock(Block):
    def __init__(self, in_ch, out_ch, time_emb_dim, **kwargs):
        super().__init__(in_ch, out_ch, time_emb_dim, **kwargs)
    def forward(self, x, t=None):
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        if t is not None:
            time_emb = self.relu(self.time_mlp(t))
            # Extend last 2 dimensions
            time_emb = time_emb[(..., ) + (None, ) * 2]
            # Add time channel
            h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)
        

    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        # down_channels = down_channels[:3]
        up_channels = (1024, 512, 256, 128, 64)
        # up_channels = up_channels[2:]
        out_dim = 2
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        # print("tst", x.shape)
        # ic()
        # print_distrib(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            # print("tst", x.shape)
            # ic()
            # print_distrib(x)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
            # ic()
            # print_distrib(x)
            # print("tst", x.shape)
        output = self.output(x)
        ic()
        print_distrib(x, "output")
        # print(f"output.shape = {output.shape}")
        return output
class SimpleCondUnet(SimpleUnet):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.conv0_cond = nn.Conv2d(1, down_channels[0], 3, padding=1)

        # Downsample
        downs = [Block(down_channels[0], down_channels[1], time_emb_dim)]
        downs += [CondBlock(down_channels[i], down_channels[i+1], \
                        time_emb_dim, scale=2) \
                    for i in range(1, len(down_channels)-1)]
        self.downs = nn.ModuleList(downs)
        self.cond_downs = nn.ModuleList([CondBlock(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        ups = [Block(up_channels[0], up_channels[1], \
                                        time_emb_dim, up=True, scale=1.5)]
        ups += [Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True, scale=1) \
                    for i in range(1, len(up_channels)-1)]
        self.ups = nn.ModuleList(ups)

    def forward(self, x, timestep):
        x_l, _ = split_lab(x)
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        cond_emb = self.conv0_cond(x_l)
        residual_inputs = []
        for down, cond_down in zip(self.downs, self.cond_downs):
            x = down(x, t)
            cond_emb = cond_down(cond_emb)
            residual_inputs.append(x)
            x = torch.cat((cond_emb, x), dim=1)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        output = self.output(x)
        return output