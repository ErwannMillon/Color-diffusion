import torch
import einops
trans = torch.nn.Linear(512, 100)
x = torch.ones(8, 512, 8, 8)
v = einops.rearrange(x, "b c h w -> b (h w) c")
t = trans(v)
print(t.shape)