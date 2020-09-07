import numpy as np
import torch


object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

"""
# get the categories for single-target
def encode_labels(target):
    k = np.zeros(len(object_categories))
    j = []

    for i in target:
        cate, diff = i
        if int(diff) == 0:
            j.append(object_categories.index(str(cate)))
    k = np.array(j[0])
    return torch.from_numpy(k)
"""


# get the categories for multiple-target
def encode_labels(target):
    k = np.zeros(len(object_categories))
    j = []

    for i in target:
        cate, diff = i
        j.append(object_categories.index(str(cate)))
        # if int(diff) == 0:
        #     j.append(object_categories.index(str(cate)))
    k[j] = 1

    return torch.from_numpy(k)

