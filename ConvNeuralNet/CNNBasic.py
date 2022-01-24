
#################################################################

from pkgutil import get_data
import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.optim import SGD, Adam
from torch import optim
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

folder = 'C:/Users/Epsilon/Desktop/images/FMNIST'

fmnist = datasets.FashionMNIST(folder, download=True, train=True)
train_images = fmnist.data
train_targets = fmnist.targets

fmnist = datasets.FashionMNIST(folder, download=True, train=False)
valid_images = fmnist.data
valid_targets = fmnist.targets


########################### Data Part ##############################

class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float().view(-1,1,28,28)/255
        self.x = x.cuda()
        self.y = y.cuda()
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return len(self.x)

def get_dataset(): # datasets : train + validation 
    train_set = FMNISTDataset(train_images, train_targets)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)  # Tuning 2. Batch Size
    valid_set = FMNISTDataset(valid_images, valid_targets)
    valid_loader = DataLoader(valid_set, batch_size=len(valid_images), shuffle=True)
    return train_loader, valid_loader


############################## Model Part ############################

class convNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.layer1 = nn.Linear(3200, 256)
        self.layer2 = nn.Linear(256, 10)

    def forward(self,x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = self.flat(x)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def get_model():
    model = convNeuralNet().cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(   # Tuning 4. Learning Rate Scheduling
        optimizer, factor=0.5, patience=0, threshold=0.001,
        verbose=True, min_lr=2e-5, threshold_mode='abs')
    return model, loss_fn, optimizer, scheduler


############################ Train Part ##############################

def train(x, y, model, optimizer, loss_fn): 
    model.train()
    loss = loss_fn(model(x),y)
    loss.backward()
    optimizer.step() # optimize(learn)
    optimizer.zero_grad()
    return loss.item()


########################## Test Part ############################

@torch.no_grad()
def validation(x, y, model, loss_fn):
    model.eval()
    loss = loss_fn(model(x),y) # calculate loss
    return loss.item()

@torch.no_grad()
def test(x, y, model):
    model.eval()
    max_values, argmaxes = model(x).max(-1) # choose class
    result = (argmaxes == y)                # result(O/X)
    return result.cpu().numpy().tolist()


######################### Put it all together ##########################

train_loader, valid_loader = get_dataset()
model, loss_fn, optimizer, scheduler = get_model() 
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []

for epoch in range(20): 
    print(epoch)
    train_batch_losses, train_batch_results = [], []
    valid_batch_losses, valid_batch_results = [], []
    
    for i, batch in enumerate(iter(train_loader)): # train loss & optimize(learn)
        x, y = batch
        train_batch_loss = train(x, y, model, optimizer, loss_fn)
        train_batch_losses.append(train_batch_loss)
    train_loss = np.mean(train_batch_losses)
    train_losses.append(train_loss)

    for i, batch in enumerate(iter(train_loader)): # train accuracy
        x, y = batch
        train_batch_result = test(x, y, model)
        train_batch_results.extend(train_batch_result)
    train_accuracy = np.mean(train_batch_results)
    train_accuracies.append(train_accuracy)

    for i, batch in enumerate(iter(valid_loader)): # validation loss
        x, y = batch
        valid_batch_loss = validation(x, y, model, loss_fn)
        valid_batch_losses.append(valid_batch_loss)
    valid_loss = np.mean(valid_batch_losses)
    scheduler.step(valid_loss)
    valid_losses.append(valid_loss)

    for i, batch in enumerate(iter(valid_loader)): # validation accuracy
        x, y = batch
        valid_batch_result = test(x, y, model)
        valid_batch_results.extend(valid_batch_result)
    valid_accuracy = np.mean(valid_batch_results)
    valid_accuracies.append(valid_accuracy)


######################## Show Result ###########################

epochs = np.arange(20)+1

plt.subplot(211)
plt.plot(epochs, train_losses, 'bo', label='Training Loss')
plt.plot(epochs, valid_losses, 'ro', label="Validation Loss")
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid('on')
plt.legend()

plt.subplot(212)
plt.plot(epochs, train_accuracies, 'bo', label='Training Accuracies')
plt.plot(epochs, valid_accuracies, 'ro', label="Validation Accuracies")
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.grid('on')
plt.legend() 

plt.show()