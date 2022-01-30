
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models, datasets
import numpy as np
import pandas as pd
import cv2, time
import matplotlib.pyplot as plt

FOLDER = 'C:/Users/Epsilon/Desktop/images/fairface/'
train_chart = pd.read_csv(FOLDER+'fairface-labels-train.csv') # chart includes images, targets
valid_chart = pd.read_csv(FOLDER+'fairface-labels-val.csv')
WIDTH, HEIGHT = 224, 224



############################## Data Part ##################################

class GenderAge(Dataset):    
    def __init__(self, chart):
        self.chart = chart
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    def __getitem__(self,i):
        data = self.chart.iloc[i].squeeze() # ith row in chart
        age, gen = (data.age), (data.gender=='Female')
        img = cv2.imread(FOLDER+data.file)
        img = self.process_image(img)
        age = torch.tensor(float(int(age)/80)).float().cuda()
        gen = torch.tensor(float(gen)).float().cuda()
        return img, gen, age

    def process_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img = torch.tensor(img).permute(2,0,1).cuda()/225.0
        img = self.normalize(img)
        return img
    
    def __len__(self):
        return len(self.chart)


def get_dataset(): 
    train_set = GenderAge(train_chart)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
    valid_set = GenderAge(valid_chart)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=True)
    return train_loader, valid_loader



######################### Model Part ###############################

def get_model():
    model = models.vgg16(pretrained=True) # Transfer Learning using pretrained VGG16
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.Sequential( # Edit final pooling after CNN
        nn.Conv2d(512, 512, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    model.classifier = GenderAgeClassifier() # Use our own classifier (defined below)
    loss_fn_gen = nn.BCELoss()
    loss_fn_age = nn.L1Loss()
    loss_fns = loss_fn_gen, loss_fn_age
    optimizer = Adam(model.parameters(), lr=1e-4)
    return model.cuda(), loss_fns, optimizer


class GenderAgeClassifier(nn.Module): # Define Multi-Classifier
    def __init__(self):
        super(GenderAgeClassifier, self).__init__()
        self.branch = nn.Sequential( # Before Seperation
            nn.Linear(2048,512), 
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512,128), 
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128,64), 
            nn.ReLU()
        )
        self.age_classifier = nn.Sequential( # Age Part
            nn.Linear(64,1), nn.Sigmoid()
        )
        self.gender_classifer = nn.Sequential( # Gender Part
            nn.Linear(64,1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.branch(x)
        age = self.age_classifier(x)
        gen = self.gender_classifer(x)
        return gen, age



####################### Train Part #########################

def train(data, model, optimizer, loss_fns):
    model.train()
    optimizer.zero_grad()
    x, y_gen, y_age = data
    pred_gen, pred_age = model(x)
    pred_gen = pred_gen.squeeze()
    pred_age = pred_age.squeeze()

    loss_fn_gen, loss_fn_age = loss_fns
    loss_gen = loss_fn_gen(pred_gen, y_gen)
    loss_age = loss_fn_age(pred_age, y_age)
    loss = loss_gen + loss_age

    loss.backward()
    optimizer.step()
    return loss.item()



######################### Test Part #########################

@torch.no_grad()
def validation(data, model, loss_fns):
    model.eval()
    x, y_gen, y_age = data
    pred_gen, pred_age = model(x)
    pred_gen = pred_gen.squeeze()
    pred_age = pred_age.squeeze()

    loss_fn_gen, loss_fn_age = loss_fns
    loss_gen = loss_fn_gen(pred_gen, y_gen)
    loss_age = loss_fn_age(pred_age, y_age)
    loss = loss_gen + loss_age

    acc_gen = ((pred_gen>0.5)==y_gen).float().sum() # sum output instead of list
    err_age = torch.abs(pred_age-y_age).float().sum()
    return loss.item(), acc_gen.item(), err_age.item()


###################### Put it all Together ########################

EPOCHS = 5
model, loss_fns, optimizer = get_model()
train_loader, valid_loader = get_dataset()
valid_accuracies_gen = []
valid_errors_age = []
train_losses, valid_losses = [], []
start = time.time()

for epoch in range(EPOCHS): # changed this summation part slightly
    train_loss, valid_loss = 0, 0 
    valid_accuracy_gen, valid_error_age = 0, 0
    length = 0

    for i, data in enumerate(train_loader): # train batch
        batch_loss = train(data, model, optimizer, loss_fns)
        train_loss += batch_loss
    
    for i, data in enumerate(valid_loader): # validation batch
        batch_loss, batch_acc_gen, batch_err_age = validation(data, model, loss_fns)
        valid_loss += batch_loss
        valid_accuracy_gen += batch_acc_gen
        valid_error_age += batch_err_age
        length += len(data[0])
    
    valid_error_age /= length
    valid_accuracy_gen /= length
    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)

    valid_accuracies_gen.append(valid_accuracy_gen)
    valid_errors_age.append(valid_error_age)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    elapsed = time.time()-start # Print TMI
    print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(epoch+1, EPOCHS, time.time()-start, (EPOCHS-epoch)*(elapsed/(epoch+1))))
    info = f'''Epoch: {epoch+1:03d}\tTrain Loss: {train_loss:.3f}\tTest: {valid_loss:.3f}\t'''
    info += f'\nGender Accuracy: {valid_accuracy_gen*100:.2f}%\tAge MAE: {valid_error_age:.2f}\n'
    print(info)


################################################################

epochs = np.arange(1,EPOCHS+1)
figure, axs = plt.subplots(2,2,figsize=(10,5))
axs[0,0].plot(epochs, valid_accuracies_gen, 'b')
axs[0,1].plot(epochs, valid_errors_age, 'r')
axs[1,0].plot(epochs, train_losses, 'g')
axs[1,1].plot(epochs, valid_losses, 'm')
axs[0,0].set_xlabel('Epochs'); axs[0,0].set_ylabel('Gender_Accuracy')
axs[0,1].set_xlabel('Epochs'); axs[0,1].set_ylabel('Age_Error')
axs[1,0].set_xlabel('Epochs'); axs[1,0].set_ylabel('Train_Loss')
axs[1,1].set_xlabel('Epochs'); axs[1,1].set_ylabel('Valie_Loss')
plt.show()