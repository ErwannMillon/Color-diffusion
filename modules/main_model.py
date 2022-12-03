# # import torch
# # from torch import nn
# # from torch import optim
# # from models import init_weights
# # # from utils import cat_lab, get_device, init_model, split_lab
# # # from unet import SimpleUnet
# # from discriminator import PatchDiscriminator, GANLoss


# # import torch.nn.functional as F



# # forward_diffusion_sample(x, 2)
#     # quit()

# # Define beta schedule
# # T = 300
# # betas = linear_beta_schedule(timesteps=T)

# # # Pre-calculate different terms for closed form
# # alphas = 1. - betas
# # alphas_cumprod = torch.cumprod(alphas, axis=0)
# # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# # sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
# # sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# # sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
# # posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# class MainModel(nn.Module):
#     def __init__(self, net_G=None, lr_G=0.001, lr_D=2e-4, 
#                  beta1=0.5, beta2=0.999, lambda_L1=100.):
#         super().__init__()
        
#         self.device = get_device()
#         self.lambda_L1 = lambda_L1
        
#         # if net_G is None:
#         #     self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
#         # else:
#         #     self.net_G = net_G.to(self.device)
#         self.unet = SimpleUnet()
#         self.unet = init_weights(self.unet, init="kaiming", leakyslope=0.5)
#         # self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
#         # self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
#         self.L1criterion = nn.L1Loss()
#         self.opt_unet = optim.Adam(self.unet.parameters(), lr=lr_G)
#         # self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
#     def set_requires_grad(self, model, requires_grad=True):
#         for p in model.parameters():
#             p.requires_grad = requires_grad
        
#     def setup_input(self, data):
#         self.L = data[:, :1, ...].to(self.device)
#         self.ab = data[:, 1:, ...].to(self.device)
#         # self.ab = data['ab'].to(self.device)
#         # print(f"self.L.shape = {self.L.shape}")
#         # print(f"self.ab.shape = {self.ab.shape}")

#     def forward(self, data, timesteps):
        
#         # print("fwd")
#         # # print(data.device)
#         # print(timesteps.device)
#         # x_l, x_ab = split_lab(data)
#         # batch = cat_lab(x_l.detach(), x_ab)
#         self.color_noise_pred = self.unet(data, timesteps)
#         # fake_image = torch.cat([x_l, self.color_noise_pred], dim=1)
#         return(self.color_noise_pred, torch.tensor([1]))
#         # self.pred = 
    
#     def backward_D(self, fake_image):
#         # fake_image = torch.cat([self.L, self.fake_color], dim=1)
#         fake_preds = self.net_D(fake_image.detach())
#         self.loss_D_fake = self.GANcriterion(fake_preds, False)
#         real_image = torch.cat([self.L, self.ab], dim=1)
#         real_preds = self.net_D(real_image)
#         self.loss_D_real = self.GANcriterion(real_preds, True)
#         self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
#         self.loss_D.backward()
    
#     def backward_unet(self, color_noise, noise, fake_image=None):
#         # fake_image = torch.cat([self.L, self.fake_color], dim=1)
#         fake_preds = self.net_D(fake_image)
#         # self.loss_G_GAN = self.GANcriterion(fake_preds, True)
#         # self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
#         # self.loss_G = self.loss_G_GAN + self.loss_G_L1
#         # self.loss_G = F.l1_loss(fake_i)
#         self.loss_G.backward()
#         return (fake_image)
    
#     def optimize(self, noise_pred, real_noise):
#         # fake_color = self.forward(data, t)
#         # fake_image = torch.cat([self.L, self.color_noise_pred], dim=1)
#         # self.net_D.train()
#         # self.set_requires_grad(self.net_D, True)
#         # self.opt_D.zero_grad()
#         # self.backward_D(fake_image)
#         # self.opt_D.step()
        
#         # self.unet.train()
#         # self.set_requires_grad(self.net_D, False)
#         self.opt_unet.zero_grad()
#         loss = F.l1_loss(real_noise, noise_pred)
#         loss.backward()
#         self.opt_unet.step()
#         return(loss)
#         # losses = {"unet_gan": self.loss_G_GAN, "unet_L1": self.loss_G_L1, 
#         #             "gen": self.loss_G, "disc": self.loss_D}
#         # return (fake_image, fake_color, losses)
