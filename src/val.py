import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from celeba_dataset import CelebADataset
from model import Net
import numpy as np


def rmse(f,g):
    return np.sqrt(np.sum(np.square(f-g))/(np.prod(f.shape)))

print("CUDA Available: ",torch.cuda.is_available())
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

val_folder = '../data/val'
batch_size = 32

val_set = CelebADataset(root_dir=val_folder, factor=2, n_samples=2000, interpolation=2)
val_loader = DataLoader(val_set, batch_size=batch_size)

net = Net()
net.load_state_dict(torch.load("weights/weights_1em3.pth", map_location='cpu'))
net.eval()


for i, (inputs, targets) in enumerate(val_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net(inputs)

    orig = targets.numpy()
    lr = inputs.numpy()
    hr = outputs.numpy()

    print(i, orig.shape, lr.shape, hr.shape)