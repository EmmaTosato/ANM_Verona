# models.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from monai.networks.nets import DenseNet121

class ResNet3D(nn.Module):
    def __init__(self, n_classes, in_channels=1):
        super(ResNet3D, self).__init__()
        self.model = r3d_18(weights=None)
        self.model.stem[0] = nn.Conv3d(in_channels, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

    def forward(self, x):
        return self.model(x)



class DenseNet3D(nn.Module):
    def __init__(self, n_classes):
        super(DenseNet3D, self).__init__()
        self.model = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes,
            pretrained=False
        )

    def forward(self, x):
        return self.model(x)

