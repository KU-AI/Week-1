import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def create_dataset(batch_size=32, data_path='data/', shuffle=True):
    trans = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_set = datasets.ImageFolder(root=data_path, transform=trans)
    train_data, val_data, test_data = random_split(data_set, [2000, 500, 500], generator=None)
    train_set = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_set = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)
    test_set = DataLoader(test_data, batch_size=batch_size)
    return train_set, val_set, test_set


# if __name__ == '__main__':
#     train_set, val_set, test_set = create_dataset(batch_size=16)
#     labels = {0: 'cat', 1: 'dog', 2: 'panda'}
#     for i, data in enumerate(val_set):
#         img, _ = data
#         plt.figure(figsize=(15, 15))
#         for j in range(len(img)):
#             plt.subplot(4, 4, j + 1)
#             plt.xticks([])
#             plt.yticks([])
#             plt.imshow(img[j].permute(1, 2, 0).numpy())
#             plt.xlabel(labels[int(_[j])])
#         plt.show()
#         break
