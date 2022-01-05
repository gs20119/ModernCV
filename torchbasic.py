
import torch
import time
import numpy as np
import torch.nn.functional as F # activation & loss
import torch.nn as nn # layer
import matplotlib.pyplot as plt # graph
from torch.utils.data import Dataset, DataLoader

x = torch.tensor([[2., -1.], [1., 1.]], requires_grad=True) # gradient is to be prepared
out = x.pow(2).sum()
out.backward() # get gradient of 
print(x.grad)
print("")

x = torch.rand(10000,6400)
y = torch.rand(6400,10000)

x, y = x.cuda(), y.cuda()
start = time.time()
#z = (x @ y)
end = time.time()
print("GPU TIME : ",f"{end - start:.5f} sec")

x, y = x.cpu(), y.cpu()
start = time.time()
#z = (x @ y)
end = time.time()
print("CPU TIME : ",f"{end - start:.5f} sec")


##########################################################

class MyNet(nn.Module): # construct network

    def __init__(self):
        super().__init__() # 1. super init
        self.layer1 = nn.Linear(2,8) # 2. define layers
        self.layer2 = nn.Linear(8,1) 
    
    def forward(self, x): 
        x = self.layer1(x) # 3. forward propagation
        x = F.relu(x)
        x = self.layer2(x)
        return x

##########################################################    

class MyDataset(Dataset): # manage numpy data
    def __init__(self, x, y):
        self.x = torch.tensor(x).float().cuda()
        self.y = torch.tensor(y).float().cuda()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

##########################################################

X = [[1,2],[3,4],[5,6],[7,8]] # data
Y = [[3],[7],[11],[15]] 
dataset = MyDataset(X,Y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

mynet = MyNet().cuda() # network 
print(mynet.layer1.weight) 
print(mynet.parameters) 

loss_history = [] 
optimizer = torch.optim.SGD(mynet.parameters(), lr=1e-3) # optimizer

##########################################################

start = time.time()
for epoch in range(50): # training 
    for data in dataloader: # 2 data in loader(epoch)
        x,y = data 
        loss = F.mse_loss(mynet(x),y) # forward
        optimizer.zero_grad() # backward
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach().cpu().numpy()) 
    # detach : copy tensor out of computational graph
end = time.time()
print(end-start)

##############################################################

x_test = [[10,11],[5,6]]
x_test = torch.tensor(x_test).float().cuda()
print(mynet(x_test).detach().cpu().numpy())

##############################################################

plt.plot(loss_history)
plt.title('Loss variation')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

###############################################################

# torch.save(model.cpu().state_dict(), 'mymodel.pth')
# state = torch.load('mymodel.pth')
# mynet.load_state_dict(state)
