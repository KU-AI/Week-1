import os
import tarfile
import collections
from torchvision.datasets.vision import VisionDataset
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data.dataset import Dataset 
# from .utils import download_url, check_integrity, verify_str_arg


class VOCclassification(Dataset):
    ## transform=>transformations, target_tranform=>encode_label
    def __init__(self, root, image_set, transform, target_transform):

        self.image_set=image_set
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        image_dir=os.path.join(self.root,'JPEGImages')
        annotation_dir = os.path.join(self.root,'Annotations')

        splits_dir=os.path.join(self.root,'ImageSets/Main')
        splits_f=os.path.join(splits_dir,image_set + '.txt')

        with open(os.path.join(splits_f), 'r') as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images=[os.path.join(image_dir,x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]

        assert(len(self.images) == len(self.annotations))

    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())
        
        img = self.transform(img)
        target = self.target_transform(target)

        return img, target

    
    def __len__(self):
        return len(self.images)
        

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict




        
            

        

