
###########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import matplotlib.pyplot as plt
from matplotlib import ticker
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import numpy as np
from random import shuffle, seed
import glob 
from glob import glob
import cv2

folder = 'C:/Users/Epsilon/Desktop/images/cat-and-dog/'
train_data_dir = folder+'training_set/training_set'
test_data_dir = folder+'test_set/test_set'


################################### Data Part ###################################

class CatsDogs(Dataset):
    def __init__(self, dir):
        cats = glob(dir+'/cats/*.jpg')
        dogs = glob(dir+'/dogs/*.jpg')
        self.fpaths = cats[:500] + dogs[:500] # Cat Data 500 + Dog Data 500
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Must (Transfer Learning)
        seed(10); shuffle(self.fpaths)
        self.targets = [fpath.split('1')[-1].startswith('dog') for fpath in self.fpaths]

    def __getitem__(self, i):
        fpath = self.fpaths[i]
        target = self.targets[i]
        im = cv2.imread(fpath)[:,:,::-1] # get image from path
        im = cv2.resize(im, (224,224)) # process image to torch
        im = torch.tensor(im/255)
        im = im.permute(2,0,1)
        im = self.normalize(im)
        return im.float().cuda(), torch.tensor([target]).float().cuda() # return (x,y)

    def __len__(self): 
        return len(self.fpaths)


def get_data():
    train_set = CatsDogs(train_data_dir)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
    valid_set = CatsDogs(test_data_dir)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=True, drop_last=True)
    return train_loader, valid_loader



################################## Model Part ###################################

def get_model_vgg16(): 
    model = models.vgg16(pretrained=True) # Use pretrained model (Transfer Learning)
    for param in model.parameters():
        param.requires_grad = False # We don't train CNN Part
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1)) # Edit final pooling after CNN 
    model.classifier = nn.Sequential( # New Classifer Part (only train this)
        nn.Flatten(), 
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128,1),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model.cuda(), loss_fn, optimizer


def get_model_res18():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.fc = nn.Sequential(
        nn.Flatten(), 
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128,1),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model.cuda(), loss_fn, optimizer



################################## Train Part ###################################

def train(x, y, model, optimizer, loss_fn):
    model.train()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


################################## Test Part ######################################

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    check = (model(x) > 0.5) == y
    return check.cpu().numpy().tolist()


################################ Put it all together ##################################

EPOCHS = 10
train_loader, valid_loader = get_data()
model, loss_fn, optimizer = get_model_res18()
train_losses, train_accuracies = [], []
valid_accuracies = []

for epoch in range(EPOCHS):
    print(epoch)
    train_batch_losses, train_batch_results = [], []
    valid_batch_results = []

    for i, batch in enumerate(iter(train_loader)):
        x, y = batch
        train_batch_loss = train(x, y, model, optimizer, loss_fn)
        train_batch_losses.append(train_batch_loss)
    train_loss = np.array(train_batch_losses).mean()
    train_losses.append(train_loss)

    for i, batch in enumerate(iter(train_loader)):
        x, y = batch
        train_batch_result = accuracy(x, y, model)
        train_batch_results.extend(train_batch_result)
    train_accuracy = np.mean(train_batch_results)
    train_accuracies.append(train_accuracy)

    for i, batch in enumerate(iter(valid_loader)):
        x, y = batch
        valid_batch_result = accuracy(x, y, model)
        valid_batch_results.extend(valid_batch_result)
    valid_accuracy = np.mean(valid_batch_results)
    valid_accuracies.append(valid_accuracy)


################################################################################

epochs = np.arange(EPOCHS)+1
plt.plot(epochs, train_accuracies, 'bo', label='Training Accuracy')
plt.plot(epochs, valid_accuracies, 'r', label='Validation Accuracy')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.title('ResNet18 Accuracy with 1K Data points')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.9,1)
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.grid('on')
plt.legend()
plt.show()


