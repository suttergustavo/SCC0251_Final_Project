import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from celeba_dataset import CelebADataset
from model import Net
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

print("CUDA Available: ",torch.cuda.is_available())
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

val_folder = '../data/val'

val_set = CelebADataset(root_dir=val_folder, factor=2, n_samples=8, interpolation=2)
degr, orig = val_set.__getitem__(0)

net = Net()
net.load_state_dict(torch.load("weights/weights_1em3.pth", map_location='cpu'))
net.eval()

degr = torch.unsqueeze(degr, 0)
sr = net(degr)

sr = sr.view(orig.shape[0], orig.shape[1], orig.shape[2])
degr = degr.view(orig.shape[0], orig.shape[1], orig.shape[2])

degr, orig, sr = degr.permute(1, 2, 0), orig.permute(1, 2, 0), sr.permute(1, 2, 0)

sr = sr.detach().numpy()

plt.subplot(131)
plt.axis('off')
plt.imshow(orig)

plt.subplot(132)
plt.axis('off')
plt.imshow(degr)

plt.subplot(133)
plt.axis('off')
plt.imshow(sr)

plt.show()