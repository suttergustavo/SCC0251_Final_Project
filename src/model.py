import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
    self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 3, 3, padding=1)
   
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    return x