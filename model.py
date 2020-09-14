import torch.nn as nn
# import torch.nn.functional as F
import torch
class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 7, 2, 3)
        # Layer1
        self.conv1_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_4 = nn.Conv2d(64, 64, 3, 1, 1)

        # Layer2
        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1)

        # Layer3
        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)

        # Layer4
        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 1)

        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)

        self.linear1 = nn.Linear(512, 64)
        self.linear2 = nn.Linear(64, 3)
        self.linear3 = nn.Linear(512, 3)

        self.relu = nn.ReLU(inplace=True)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #     elif isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         torch.nn.init.constant_(m.weight, 1)
        #         torch.nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = (self.relu(self.bn64(self.conv0(x))))

        # Layer1
        x = self.max_pool(x)

        y = self.bn64(self.conv1_2(self.relu(self.bn64(self.conv1_1(x)))))
        x = x.clone() + y          # residual connection
        x = self.relu(x)

        y = self.bn64(self.conv1_4(self.relu(self.bn64(self.conv1_3(x)))))
        x = x.clone() + y          # residual connection
        x = self.relu(x)

        # Layer2
        x = self.bn128(self.conv2_2(self.relu(self.bn128(self.conv2_1(x)))))
        x = self.relu(x)

        y = self.bn128(self.conv2_4(self.relu(self.bn128(self.conv2_3(x)))))
        x = x.clone() + y          # residual connection
        x = self.relu(x)

        # Layer3
        x = self.bn256(self.conv3_2(self.relu(self.bn256(self.conv3_1(x)))))
        x = self.relu(x)

        y = self.bn256(self.conv3_4(self.relu(self.bn256(self.conv3_3(x)))))
        x = x.clone() + y          # residual connection
        x = self.relu(x)

        # Layer4
        x = self.bn512(self.conv4_2(self.relu(self.bn512(self.conv4_1(x)))))
        x = self.relu(x)

        y = self.bn512(self.conv4_4(self.relu(self.bn512(self.conv4_3(x)))))
        x = x.clone() + y          # residual connection
        x = self.relu(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)    # x = torch.flatten(x, 1)
        # x = self.relu(self.linear1(x))
        # x = self.linear2(x)
        x = self.linear3(x)

        return x
