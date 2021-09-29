from torch import flatten
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(3,3), stride=(1,1), padding=(0,0))

        self.linear1 = nn.Linear(1080, 84)
        self.linear2 = nn.Linear(84, out_channels)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.relu(x)

        x = flatten(x, start_dim=1)

        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = nn.Softmax()(x)

        return x
