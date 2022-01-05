
import torch
import numpy as np

x = torch.tensor([[1,2]]) # 1x2
print(x.shape)
print(x.dtype)

y = torch.tensor([[1],[2.0]]) # 2x1
print(y.shape)
print(y.dtype)
print("")

torch.zeros((3,4))
torch.ones((3,4))
torch.randint(low=0, high=10, size=(3,4))
torch.rand((3,4))
torch.randn((3,4))
torch.arange(25)

x = np.array([[10,20,30],[2,3,4]]) # 2x3
y = torch.tensor(x)
print(y.shape)
print(y)
print(y*10)
print(y.add(10)) # y doesn't change
print(y)
print("")

z = torch.tensor([2,3,1,0]) # 1-dimension column vector
print(z.shape) 
print(z)
z1 = z.view(1,4) # [[]] : 2-dimension
print(z1)
z = z.view(4,1)
print(z)
print("")

x = torch.randn((3,1,5))
x = x.squeeze(1) # squeeze axis-1
print(x.shape)
x = x.unsqueeze(0) # unsqueeze axis-0
print(x.shape)
print("")

x = torch.randn(10,10)
x1, x2, x3 = x[None], x[:,None], x[:,:,None]
print(x1.shape, x2.shape, x3.shape)
print("")

y = torch.tensor([[1,2,3,4],[5,6,7,8]]) # 2x4
z = torch.tensor([2,3,1,0]) # 4
z1 = torch.tensor([[2,1],[3,0],[1,1],[0,1]]) # 4x2
print(torch.matmul(y,z)) # (2x4) * (4) = 2
print(torch.matmul(y,z1)) # (2x4) * (4x2) = 2x2
print(y@z1)
print("")

x = torch.randn(2,2,2)
y = torch.randn(1,2,2)
z = torch.cat([x,y], axis=0)
print(z)


