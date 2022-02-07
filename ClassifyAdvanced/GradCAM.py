
#################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_snippets
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
import cv2, time
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

idx = { 'Parasitized':0, 'Uninfected':1 }
files = glob('C:/Users/Epsilon/Desktop/images/cell_images/*/*.png') # glob directories
np.random.seed(10)
np.random.shuffle(files)
train_files, valid_files = train_test_split(files, random_state=1)



######################### Data Part ###############################

train_preprocess = T.Compose([  # implement image processing with T.Compose
    T.ToPILImage(),
    T.Resize(128),
    T.CenterCrop(128),
    T.ColorJitter(
        brightness=(0.95,1.05),
        contrast=(0.95,1.05),
        saturation=(0.95,1.05),
        hue=0.05
    ),
    T.RandomAffine(5,translate=(0.01,0.1)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

valid_preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


class Malaria(Dataset):
    def __init__(self, files, prep=None):
        self.files = files
        self.preprocess = prep
        
    def __getitem__(self, i):
        fpath = self.files[i]
        img_raw = cv2.imread(fpath)
        img_new = self.preprocess(img_raw)
        targ = fpath.split('\\')[-2]    # Parasitized vs Uninfected
        targ = torch.tensor([idx[targ]])
        return img_new.cuda(), targ.cuda()

    def __len__(self):
        return len(self.files)


def get_data():
    train_set = Malaria(train_files, prep=train_preprocess)
    valid_set = Malaria(valid_files, prep=valid_preprocess)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)
    return train_loader, valid_loader



############################### Model Part #####################################

class MalariaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            self.convBlock(3,64), # initial channels = 3 (RGB)
            self.convBlock(64,64),
            self.convBlock(64,128),
            self.convBlock(128,256),
            self.convBlock(256,512),
            self.convBlock(512,64),
            nn.Flatten(), # Flatten to 256-vector
            nn.Linear(256,256),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256,2) # result - two numbers (likelihood)
        )

    def forward(self, x):
        return self.model(x)

    def convBlock(self, nin, nout):
        return nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(nin, nout, kernel_size=3, padding=1), # filter size = 3x3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nout),
            nn.MaxPool2d(2)
        )


def get_model():
    model = MalariaClassifier().cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer



################################ Train Part #####################################

def train(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    x, y = data  
    y = y.squeeze()
    pred = model(x)
    loss = loss_fn(pred,y)
    loss.backward()
    optimizer.step()
    acc = (torch.max(pred,1)[1]==y).float().mean()
    return loss.item(), acc.item()



################################ Test Part #####################################

@torch.no_grad()
def validation(model, data, loss_fn):
    model.eval()
    x, y = data
    y = y.squeeze()
    pred = model(x)
    loss = loss_fn(pred,y)
    acc = (torch.max(pred,1)[1]==y).float().mean()
    return loss.item(), acc.item()



############################ Put it all together ###################################

EPOCHS = 3
model, loss_fn, optimizer = get_model()
train_loader, valid_loader = get_data()
x, y = next(iter(valid_loader))
train_losses, valid_losses = [], []
train_accuracies, valid_accuracies = [], []
start = time.time()

for epoch in range(EPOCHS):
    train_loss, valid_loss = 0, 0 
    train_accuracy, valid_accuracy = 0, 0
    train_length, valid_length = 0, 0

    for i, data in enumerate(train_loader):
        batch_loss, batch_acc = train(model, data, optimizer, loss_fn)
        train_loss += batch_loss
        train_accuracy += batch_acc
        train_length += len(data[0])
    
    for i, data in enumerate(valid_loader):
        batch_loss, valid_acc = validation(model, data, loss_fn)
        valid_loss += batch_loss
        valid_accuracy += batch_acc
        valid_length += len(data[0])

    train_loss /= train_length
    valid_loss /= valid_length
    train_accuracy /= len(train_loader)
    valid_accuracy /= len(valid_loader)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)

    elapsed = time.time()-start # Print TMI
    print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(epoch+1, EPOCHS, time.time()-start, (EPOCHS-epoch)*(elapsed/(epoch+1))))
    info = f'''Epoch: {epoch+1:03d}\tTrain Loss: {train_loss:.3f}\tTest: {valid_loss:.3f}\t'''
    info += f'\nValid Accuracy: {valid_accuracy*100:.2f}%\tTrain Accuracy: {train_accuracy*100:.2f}%\n'
    print(info)



######################################################################################

epochs = np.arange(EPOCHS)+1

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
plt.grid('on')
plt.legend() 

plt.show()



#################################### Generating CAM #####################################

subModel = nn.Sequential(*(list(model.model[:5].children())+list(model.model[5][:2].children())))

def generateCAM(x): # generate Gradient based CAM
    model.eval()
    pred = model(x)
    heatmap = []
    decisn = pred.max(-1)[-1]
    activations = subModel(x)
    print(activations.shape)
    model.zero_grad()
    pred[0,decisn].backward(retain_graph=True)
    pooled_grads = model.model[-6][1].weight.grad.data.mean((1,2,3))
    for i in range(activations.shape[1]):
        activations[:,i,:,:] *= pooled_grads[i] # weight with gradient
    heatmap = torch.mean(activations, dim=1)[0].cpu().detach()
    return heatmap, 'Uninfected' if decisn.item() else 'Parasitized'

def processHeatmap(map, img): # edit heatmap + image for visualizing
    m, M = map.min(), map.max()
    map = 255*((map-m)/(M-m))
    map = np.uint8(map)
    map = cv2.resize(map, (128,128))
    map = cv2.applyColorMap(255-map, cv2.COLORMAP_JET)
    map = np.uint8(map)
    map = np.uint8(map*0.7+img*0.3)
    return map


SAMPLES = 10
x, y = next(iter(train_loader))

for i in range(SAMPLES):
    img_raw = x[i].permute(1,2,0).cpu().numpy() 
    heatmap, decisn = generateCAM(x[i:i+1])
    if(decisn=='Uninfected'): continue
    heatmap = processHeatmap(heatmap, img_raw)
    torch_snippets.subplots([img_raw, heatmap], nc=2, figsize=(5,3), suptitle=decisn) 
    # temporary library. Need to fix this to plt.subplot