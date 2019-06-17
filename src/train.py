import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from celeba_dataset import CelebADataset
from model import Net

print("CUDA Available: ",torch.cuda.is_available())
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


train_folder = '' 
val_folder = ''
batch_size = 64
epochs = 20
learning_rate = 1e-3

train_set = CelebADataset(root_dir=train_folder, factor=2, n_samples=5000)
val_set = CelebADataset(root_dir=val_folder, factor=2, n_samples=1000)

train_loader = DataLoader(train_set, batch_size=batch_size)

net = Net().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        if i % 1000:
            print(f'{epoch} {i}: {loss.item()}')