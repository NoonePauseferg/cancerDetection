import cv2 as cv
from torch.utils.data import Dataset
import os
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch

class dataClass(Dataset):
    def __init__(self, img_labels, img_dir, transform=None):
        super().__init__()
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.img_labels.shape[0]

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0] + '.tif')
        image = cv.imread(img_path)
        label = self.img_labels.iloc[index, 1]
        if self.transform is not None:
            image = self.transform(np.transpose(image, (1,2,0)))
        return image, label
