
##################################################################

from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD

folder = 'C:/Users/Epsilon/Desktop/images/FMNIST'
fmnist = datasets.FashionMNIST(folder, download=True, train=True)
train_images = fmnist.data
train_targets = fmnist.targets

##################################################################

R, C = len(train_targets.unique()), 10
fig, ax = plt.subplots(R, C, figsize=(10,10))
for labelClass, plotRow in enumerate(ax):
    labelXRows = np.where(train_targets == labelClass)[0]
    for cell in plotRow:
        cell.grid(False)
        cell.axis('off')
        ix = np.random.choice(labelXRows)
        x, y = train_images[ix], train_targets[ix]
        cell.imshow(x, cmap='gray')

plt.tight_layout()
#plt.show()

#################################################################

class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.float().view(-1,28*28).cuda()
        self.y = y.cuda()

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)

def get_dataset():
    train_set = FMNISTDataset(train_images, train_targets)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    return train_loader

#################################################################

def get_model():
    model = nn.Sequential(
        nn.Linear(28*28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    ).cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2)
    return model, loss_fn, optimizer

###############################################################

def train_batch(x, y, model, optimizer, loss_fn):
    model.train()
    loss = loss_fn(model(x),y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

###############################################################

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    max_values, argmaxes = model(x).max(-1)
    valid = (argmaxes == y)
    return valid.cpu().numpy().tolist()

###############################################################

train_loader = get_dataset()
model, loss_fn, optimizer = get_model()
losses, accuracies = [], []

for epoch in range(5): 
    print(epoch)
    epoch_losses, epoch_accuracies = [], []
    
    for i, batch in enumerate(iter(train_loader)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)
        epoch_losses.append(batch_loss)
        valid = accuracy(x, y, model)
        epoch_accuracies.append(valid)

    epoch_loss = np.mean(epoch_losses)
    epoch_accuracy = np.mean(epoch_accuracies)
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracies)

###############################################################




