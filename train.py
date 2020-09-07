# Training code for the classification by VGGNet
import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from vgg16 import VGG
from dataset import CustomDataset
from utils import encode_labels


class config():
    def __init__(self):
        self.imgf_root = './data/VOCdevkit/VOC2012_train/JPEGImages'
        self.annotationf_root = './data/VOCdevkit/VOC2012_train/Annotations'
        self.image_setf = './data/VOCdevkit/VOC2012_train/ImageSets/Main'
        self.batch_size = 32
        self.num_workers = 4
        self.device = torch.device('cuda')
        self.learning_rate = 1e-3
        self.epoch = 50
        self.save_path = './weights/vgg16_result.pth'
        self.writer_path = SummaryWriter('training/fashion_mnist_experiment_1')

"""
mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
"""
mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]

if __name__ == '__main__':

    # Environment setting
    cfg = config()
    train_image_set_root = '{}/train.txt'.format(cfg.image_setf)
    val_image_set_root = '{}/val.txt'.format(cfg.image_setf)

    # input data pre-processing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std),
    ])

    # training/validation/test data loading
    trainset = CustomDataset(cfg.imgf_root, cfg.annotationf_root, train_image_set_root,
                             transform = transform, target_transform = encode_labels)

    train_loader = DataLoader(trainset, batch_size=cfg.batch_size,
                              num_workers=cfg.num_workers, shuffle=True)

    # training/validation/test data loading
    valset = CustomDataset(cfg.imgf_root, cfg.annotationf_root, val_image_set_root,
                             transform = transform, target_transform = encode_labels)

    val_loader = DataLoader(valset, batch_size=cfg.batch_size,
                             num_workers=cfg.num_workers, shuffle=False)

    # CNN model - vgg16
    model = VGG()
    model.to(cfg.device)

    # loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)

    # train
    for epoch in range(cfg.epoch):
        model.train(True)
        running_loss = 0.0

        for i, data in enumerate(test_loader,0):
            img, target = data
            target = target.long()
            img, target = img.to(cfg.device), target.to(cfg.device)
            optimizer.zero_grad()
            
            outputs = model(img)

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f'%(epoch+1, i+1, running_loss/50))
                running_loss = 0.0
        torch.save(model.state_dict(), cfg.save_path)
    
    print('Finished Training')
