import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from celeba_dataset import CelebADataset
from model import Net
from torchvision.utils import make_grid
import numpy as np
import imageio
import matplotlib.pyplot as plt


def rmse(f,g):
    return np.sqrt(np.sum(np.square(f-g))/(np.prod(f.shape)))

print("CUDA Available: ",torch.cuda.is_available())
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

val_folder = '../data/val'

val_set = CelebADataset(root_dir=val_folder, factor=2, n_samples=8, interpolation=2)
degr, orig = val_set.__getitem__(img)

net = Net()
net.load_state_dict(torch.load("weights/weights_1em3.pth", map_location='cpu'))
net.eval()

degr = torch.unsqueeze(degr, 0)
sr = net(degr)

sr = sr.view(orig.shape[0], orig.shape[1], orig.shape[2])
degr = degr.view(orig.shape[0], orig.shape[1], orig.shape[2])

degr, orig, sr = degr.permute(1, 2, 0), orig.permute(1, 2, 0), sr.permute(1, 2, 0)

sr = sr.detach().numpy()
degr = degr.numpy()
orig = orig.numpy()

degr_rmse = rmse(orig, degr)
sr_rmse = rmse(orig, sr)

plt.figure(figsize=(100, 50))

plt.subplot(131)
plt.axis('off')
plt.title('Original')
plt.imshow(orig)

plt.subplot(132)
plt.axis('off')
plt.title(f'Low Resolution (RMSE: {degr_rmse:.4f})')
plt.imshow(degr)

plt.subplot(133)
plt.axis('off')
plt.title(f'Super Resolution (RMSE: {sr_rmse:.4f})')
plt.imshow(sr)

plt.show()