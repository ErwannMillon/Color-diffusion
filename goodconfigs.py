
colordiff_config = dict(
    device = "gpu",
    pin_memory = True,
    T=350,
    # lr=6e-4,
    lr = 3e-4,
    loss_fn = "l2",
    batch_size=72,
    accumulate_grad_batches=1,
    img_size = 64,
    sample=True,
    should_log=True,
    epochs=200,
    using_cond=True,
    display_every=100,
    dynamic_threshold=False,
    train_autoenc=False,
    enc_loss_coeff = 1.1,
) 

unet_config = dict(
    channels=3,
    dropout=0.3,
    self_condition=False,
    out_dim=2,
    dim=64,
    condition=True,
    dim_mults=[1, 2, 4, 8],
)
enc_config = dict(
    channels=1,
    dropout=0.3,
    self_condition=False,
    out_dim=2,
    dim=64,
    dim_mults=[1, 2, 4, 8],
)
