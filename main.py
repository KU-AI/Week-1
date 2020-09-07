import torch
import torchvision.models as models
import torch.optim as optim
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_data = datasets.ImageFolder(root='/home/lbs/Downloads/VOC_train/', transform=transform)
test_data = datasets.ImageFolder(root='/home/lbs/Downloads/VOC_test/', transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = models.vgg16(pretrained=False)
# net = models.resnet50(pretrained=False)

print(net)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=0.00001)

for epoch in range(3):  # loop over the dataset multiple times
    running_loss = 0.0

    if(epoch>0):
        net.load_state_dict(torch.load(save_path))
        net.to(device)

    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs, f = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if(loss.item() > 1000):
            print(loss.item())
            for param in net.parameters():
                print(param.data)
        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

    save_path="/vgg16_1758.pth"
    torch.save(net.state_dict(), save_path)

print('Finished Training')

class_correct = list(0. for i in range(1000))
class_total = list(0. for i in range(1000))

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs,_ = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(16):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

accuracy_sum=0
for i in range(1000):
    temp = 100 * class_correct[i] / class_total[i]
    print('Accuracy of ' + str(temp))
    accuracy_sum+=temp
print('Accuracy average: ', accuracy_sum/1000)
