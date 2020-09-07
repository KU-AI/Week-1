import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    """
    imgf_root, annotationf_root: image & annotation folder root
    image_set_root: txt file path for type of image set (ex, '././train.txt')
    transform: image transformation
    target_transform: label transformation
    """
    def __init__(self, imgf_root, annotationf_root, image_set_root,
                 transform, target_transform):
        self.imgf_root = imgf_root
        self.annotationf_root = annotationf_root
        self.image_set_root = image_set_root
        self.transform = transform
        self.target_transform = target_transform

        with open(self.image_set_root, "r") as f:
            file_names = [x.rstrip() for x in f]
        self.images = ['{}/{}.jpg'.format(self.imgf_root,x) for x in file_names]
        self.annotations = ['{}/{}.xml'.format(self.annotationf_root,x) for x in file_names]

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        target = ET.parse(self.annotations[idx]).getroot()
        target = self.find_object_list(target)
        img = self.transform(img)
        target = self.target_transform(target)
        return img, target

    def find_object_list(self, target):
        objects = target.findall("object")
        object_list = []
        for obj in objects:
            obj_name = obj.find("name").text
            obj_difficult = obj.find("difficult").text
            object_list.append([obj_name, obj_difficult])
        return object_list

    def __len__(self):
        return len(self.images)
