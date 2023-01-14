from omegaconf import OmegaConf
colordiff_config = dict(
    device = "gpu",
    pin_memory = True,
    T=350,
    # lr=6e-4,
    lr = 1e-6,
    loss_fn = "l2",
    batch_size=36,
    accumulate_grad_batches=2,
    img_size = 64,
    sample=True,
    should_log=True,
    epochs=14,
    using_cond=True,
    display_every=2,
    dynamic_threshold=False,
    train_autoenc=False,
    enc_loss_coeff = 1.1,
) 

inference_config = colordiff_config
inference_config["batch_size"] = 1


unet_config = dict(
    channels=3,
    dropout=0.3,
    self_condition=False,
    out_dim=2,
    dim=128,
    condition=True,
    dim_mults=[1, 2, 3, 3],
)
enc_config = dict(
    channels=1,
    dropout=0.3,
    self_condition=False,
    out_dim=2,
    dim=128,
    dim_mults=[1, 2, 3, 3],
)
conf = OmegaConf.create(unet_config)
print(OmegaConf.to_yaml(conf))
def load_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf
# def loadprint(load_config('./configs/encoder_config.yaml'))