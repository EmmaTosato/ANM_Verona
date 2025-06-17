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


class VGG16_3D(nn.Module):
    def __init__(self, num_classes=2, input_channels=1):
        super(VGG16_3D, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Output of features has shape: (batch_size, 512, 2, 3, 2)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 3 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten all but batch
        x = self.classifier(x)
        return x
