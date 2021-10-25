import torch
from torch import nn
import torch.nn.functional as F

class RotationalTransformer(nn.Module):
    def __init__(self, model):
        super(RotationalTransformer, self).__init__()
        
        # CNN Classifier
        self.net = model

        # network to approximate rotation angle (self.theta)
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True))
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 1))

    
        
        # self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def rotate(self, x):

        # Forward Prop Image --> Get Theta
        xs = self.localization(x)
        xs = torch.flatten(xs, start_dim=1)
        theta = self.fc_loc(xs)

        # Build Affine Transform, rotates image by angle theta
        rotate_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], 
                                      [torch.sin(theta),  torch.cos(theta), 0]], dtype=torch.float32)
        
        # Apply rotation
        grid = F.affine_grid(rotate_matrix.unsqueeze(0), x.size())
        x = F.grid_sample(x, grid)
        
        return x

    def forward(self, x):
        # rotate the input
        x = self.rotate(x)
        
        # normal forward pass
        x = self.net(x)

        return x
 



# TEST 

if __name__ == "__main__":
    from lenet import LeNet
    net = LeNet(1, 10)
    rt = RotationalTransformer(net)

    img = torch.rand(1, 1, 24, 24)

    print(rt.rotate(img).shape)
    print(net(img))
    print(rt(img))
