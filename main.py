from rotational_transformer import RotationalTransformer
from spatial_transformer import SpatialTransformer
from lenet import LeNet as Net

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn

root = '../../data'
epochs = 30
batch_size = 16
lr = .01



# Data

transform = transforms.Compose([transforms.RandomAffine(
                                    degrees=(-45, 45), 
                                    scale=(0.7, 1.2),
                                    translate=(0.1, 0.3),),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),])



test_set = datasets.MNIST(root=root, train=False, transform=transform, target_transform=None, download=False)
train_set = datasets.MNIST(root=root, train=True, transform=transform, target_transform=None, download=False)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)


### MODEL
net = RotationalTransformer(Net(1,10))

# LOSS
loss_fn = nn.CrossEntropyLoss()

# OPTIMIZER
optim = torch.optim.SGD(net.parameters(), lr=lr)

## TRAIN LOOP

for i in range(epochs):
    
    # TRAIN
    net.train()
    for batch, (x, y) in enumerate(train_loader):
        optim.zero_grad()
        y_hat = net(nn.functional.pad(x, (2,2,2,2)))
        loss = loss_fn(y_hat, y)

        
        if batch % 100 == 0:
            print('Epoch: ', i + 1, '  |  Batch: ', batch, '/', len(train_loader), '  |  Loss: ', loss)

        loss.backward()
        optim.step()

    # TEST
    net.eval()
    avg_loss = 0
    correct = 0
    for batch, (x, y) in enumerate(train_loader):
        y_hat = net(nn.functional.pad(x, (2,2,2,2)))
        avg_loss += loss_fn(y_hat, y)
        for vector in range(len(y_hat)):
            if torch.argmax(y_hat[vector]) == y[vector]:
                correct += 1
        
    print('TEST:     ', ' |   AVG LOSS: ', avg_loss / len(train_loader), '   |   Accuracy: ', correct / (len(train_loader) * batch_size))