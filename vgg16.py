# Fixed model of size of stride and padding
import numpy as np
import torch
import torchvision
import torch.nn as nn

# VGGNet block
class VGG(nn.Module):
    def __init__(self, num_class = 20, init_weights = True):
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding =1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding =1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size = 3, padding =1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, padding =1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, kernel_size = 3, padding =1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding =1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding =1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, kernel_size = 3, padding =1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding =1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding =1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512, 512, kernel_size = 3, padding =1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding =1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding =1), nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.network = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.network(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
