import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from celeba_dataset import CelebADataset
from model import Net

# Checking of there is a GPU available
print("CUDA Available: ",torch.cuda.is_available())
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Specifing important things
train_folder = '../data/train'
batch_size = 64
epochs = 20
learning_rate = 1e-4

# Getting the training set and the training loader
train_set = CelebADataset(root_dir=train_folder, factor=2, n_samples=5000)
train_loader = DataLoader(train_set, batch_size=batch_size)

# Creating the network, the loss and the optimizer
net = Net().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

# Vector that will store the loss during training
loss_history = []

for epoch in range(epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())

    print(f'Epoch {epoch} - Loss: {loss.item()}')

# Saving the loss history
loss_history = np.array(loss_history)
np.save('loss_history.npy', loss_history)

# Saving the training weights
torch.save(net.state_dict(), 'weights/weights_1em4.pth')
