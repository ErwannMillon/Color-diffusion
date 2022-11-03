import torch

from utils import get_device
def add_to_tb(noise_pred, real_noise, e):
    writer.add_histogram("pred_max", noise_pred.max(), e)
    writer.add_histogram("noise pred", noise_pred, e)
    writer.add_histogram("noise real", real_noise, e)

    writer.add_histogram("pred min", noise_pred.min(), e)
    writer.add_histogram("pred mean", noise_pred.mean(), e)
    writer.add_histogram("pred std", noise_pred.std(), e)

def make_graph(model, writer, dataloader):
	device = get_device()
    t = torch.Tensor([1]).to(device).long()
    test_batch = next(dataloader)
    # summary(model, input_data=(test_batch, t), depth=5)
    writer.add_graph(model, (test_batch, t))
    writer.close()
