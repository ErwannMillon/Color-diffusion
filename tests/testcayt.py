import torch
x = torch.ones(1, 3, 3)
v = [torch.zeros_like(x)] * 2

y = torch.cat((x, *[torch.zeros_like(x)] * 2))
print(y.shape)