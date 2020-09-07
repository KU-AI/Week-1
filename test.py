# Test code for the classification by VGGNet
import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from vgg16 import VGG
from dataset import CustomDataset
from utils import encode_labels


class config():
    def __init__(self):
        self.imgf_root = './data/VOCdevkit/VOC2012_train/JPEGImages'
        self.annotationf_root = './data/VOCdevkit/VOC2012_train/Annotations'
        self.image_setf = './data/VOCdevkit/VOC2012_train/ImageSets/Main'
        self.batch_size = 16
        self.num_workers = 4
        self.device = torch.device('cuda')
        self.learning_rate = 1e-3
        self.epoch = 1
        self.save_path = './weights/vgg16_result.pth'

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
    model.load_state_dict(torch.load(cfg.save_path))
    model.to(cfg.device)
    model.eval()

    count = [0 for i in range(20)]
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            img, target = data
            target = target.float()
            img, target = img.cuda(), target.cuda()
            for idx in range(len(target)):
                for j in range(len(target[idx])):
                    if target[idx][j] == 1:
                        count[j] += 1
    print(count)



"""
    count_right = 0
    count_wrong = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            img, target = data
            target = target.long()
            img, target = img.cuda(), target.cuda()
            outputs = model(img)
            outputs = nn.Sequential(nn.Softmax())(outputs)
            for idx, vec in enumerate(outputs,0):
                vec = list(vec)
                j = vec.index(max(vec))
                k = target[idx]
                print('j: %.2i, k: %.2i' % (j, k))
                if j == k:
                    count_right += 1
                else:
                    count_wrong += 1

    print('Right: %4i, Wrong: %4i' % (count_right, count_wrong))
    print('Accuracy: %3.3f' % (100*count_right/(count_right+count_wrong)))            
"""


"""
            for idx in range(len(target)):
                a = target[idx].argmax()
                print(int(a))
                count[a] += 1
    print(count)
"""



