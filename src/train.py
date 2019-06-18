import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from celeba_dataset import CelebADataset
from model import Net

print("CUDA Available: ",torch.cuda.is_available())
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_folder = '../data/train'
batch_size = 64
epochs = 20
learning_rate = 1e-4

train_set = CelebADataset(root_dir=train_folder, factor=2, n_samples=5000)

train_loader = DataLoader(train_set, batch_size=batch_size)

net = Net().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

loss_history = []

for epoch in range(epochs):
    running_loss = 0
    count = 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        count += 1

    print(f'Epoch {epoch} - Loss: {loss.item()}')
    loss_history.append(running_loss/count)


loss_history = np.array(loss_history)
np.save('loss_history.npy', loss_history)

torch.save(net.state_dict(), 'weights/weights_1em4.pth')
