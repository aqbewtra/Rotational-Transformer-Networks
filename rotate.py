import torch
import torchvision
import PIL.Image as Image

import torchvision.transforms as transforms

five = Image.open("5.jpg")
five_ = transforms.PILToTensor()(five)

# rotate = torch.tensor([[sin()]])
