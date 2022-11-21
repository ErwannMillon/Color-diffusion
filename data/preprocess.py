from dataset import ColorizationDataset, make_dataloaders
from utils import show_lab_image


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from PIL import Image, ImageStat
    from train import config
    import glob
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn, optim
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image,ImageChops
    import csv
    paths = ["./data/pokemon.jpg"]
    with open("./train_filtered.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        train_paths = data[0]
    with open("./val_filtered.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        val_paths = data[0]
    dataset = ColorizationDataset(paths, split="train", config=config)

    for i, path in enumerate(train_paths):
        tensor = dataset.get_tensor_from_path(path)
        # print(path)
        new_path = path.replace("fairface", "preprocessed_fairface")
        new_path = new_path.replace("jpg", "pt")
        torch.save(tensor, new_path)
        # loaded = torch.load(new_path)
        # show_lab_image(tensor.unsqueeze(0), log=False)
        # show_lab_image(loaded.unsqueeze(0), log=False)
        # plt.show()
    for i, path in enumerate(val_paths):
        tensor = dataset.get_tensor_from_path(path)
        # print(path)
        new_path = path.replace("fairface", "preprocessed_fairface")
        new_path = new_path.replace("jpg", "pt")
        torch.save(tensor, new_path)
    # loaded = torch.load("./im.pt")
    # print(tensor == loaded)
    # plt.show()

