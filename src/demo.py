import sys
import torch
import torch.nn as nn
import torch.optim as optim
from model import Net
from celeba_dataset import CelebADataset
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


def rmse(f,g):
    """
    Calculates the RMSE between two images
    """
    return np.sqrt(np.sum(np.square(f-g))/(np.prod(f.shape)))

if len(sys.argv) != 2:
    print(f'usage: python3 {sys.argv[0]} IMG_PATH')
    exit(-1)

img_path = sys.argv[1]

# Loading the image passed as a argument
dataset = CelebADataset(None, 2, 1, 2, single=img_path)
degr, orig = dataset.__getitem__(0)

# Loading the network with trained weights
net = Net()
net.load_state_dict(torch.load("weights/weights_1em3.pth", map_location='cpu'))
net.eval()

# Preparing the image to the model
degr = torch.unsqueeze(degr, 0)

# Passing the image through the network
sr = net(degr)

# Changing the shape of the output to the desired way
degr = degr.view(orig.shape[0], orig.shape[1], orig.shape[2])
sr = sr.view(orig.shape[0], orig.shape[1], orig.shape[2])

# Transforming the tensor in numpy arrays
degr, orig, sr = degr.permute(1, 2, 0), orig.permute(1, 2, 0), sr.permute(1, 2, 0)

sr = sr.detach().numpy()
degr = degr.numpy()
orig = orig.numpy()

# Calculating the errors
degr_rmse = rmse(orig, degr)
sr_rmse = rmse(orig, sr)

# Showing the original image, the low resolution image and the result
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