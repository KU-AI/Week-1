import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import numpy as np
import torch
####################

class ResNet(nn.Module):
    def __init__(self, hidden_layers , inplanes, planes, num_classes=1000, init_weights=True):
        super(ResNet, self).__init__()
        self.input_layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=7, padding=1, stride=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.relu=nn.ReLU(inplace=True)
        self.hidden_layers, self.output_channels=hidden_layers
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.output_layers = nn.Sequential(
            nn.Linear(self.output_channels, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        identity = x
        for i in self.input_layers:
            x = i(x)
            identity = i(identity)
        x += identity
        identity = self.relu(identity)
        x = self.relu(x)

        for i in self.hidden_layers:
            identity = i(identity)
            x = i(x)
            x += identity
            x = self.relu(x)
            identity = self.relu(identity)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.output_layers(x)
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

def make_layers(n_layers, input_channels, output_channels):
        layers=[]
        res=[]
        layers += [nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=1)]
        input_channels=output_channels
        conv2d = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=1)
        for i in range(n_layers-1):
            layers += [conv2d, conv2d]
        res=layers
        output_channels=output_channels*2
        return layers, res, input_channels, output_channels

def make_hidden(structure):
    input_channels=64
    output_channels=64
    layers=[]
    for i, set in enumerate(structure):
        r_layers, res, input_channels, output_channels = make_layers(set, input_channels, output_channels)
        layers += r_layers
        if i == len(structure)-1:
            output_channels=int(output_channels/2)
    return nn.Sequential(*layers), output_channels

def resnet(structure):
    model = ResNet(make_hidden(structure),3,64)
    return model

structure=[3,4,6,3]
cuda = torch.device("cuda")
resnet=resnet(structure)
resnet.to('cuda')
print(resnet)


epoch=50
batch_size=50
### dataset load ###
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
for epoch in tqdm(range(epoch)):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % batch_size == batch_size-1:    # print every 500 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / batch_size))
            running_loss = 0.0
print('Finished Training')
### Saving model ###
PATH = './cifar_resnet.pth'
torch.save(resnet.state_dict(), PATH)
model_weight=torch.load(PATH)
model_weight
weights=resnet.state_dict()
### Predicting ###
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
