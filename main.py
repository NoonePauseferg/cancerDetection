from dataClass import dataClass
import pandas as pd
import cv2 as cv
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision
import numpy as np
import os

def showImage(data, num):
    a = None
    df = data.img_labels
    q = df[df['label'] == 1].sample(n=num)
    batch = q['id'].index
    for i in batch:
        if a is None:
            a = data[i][0][None, :, :, :]
        else:
            a = torch.vstack((a, data[i][0][None, :, :, :]))
    grid_img = torchvision.utils.make_grid(np.transpose(a, (0,3,2,1)),nrow=num//5,padding=5)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title('Photos with cancer')
    plt.show()

if __name__ == '__main__':

    path_testLabels, path_testImages = pd.read_csv("data/train_labels.csv"), "data/train"
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(p=0.7),
                                    transforms.RandomVerticalFlip(p=0.7)])
    test_data = dataClass(path_testLabels, path_testImages, transform=transform)
    showImage(test_data, 25)


