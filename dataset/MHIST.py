from torch.utils.data import Dataset
from torchvision import datasets
import matplotlib as plt
import numpy as np
from PIL import Image
import pandas
import os

MHIST_LABELS = {
    "HP": 0,
    "SSA": 1
}

class MHIST(Dataset):
    def __init__(self, path, transform=None):
        self.file_path = path
        self.dataset_x = None
        self.dataset_y = None
        self.transform = transform

        # read meta data
        for file in os.listdir(self.file_path):
            if file.endswith(".csv"):
                meta_data = pandas.read_csv(self.file_path + '/' + file)
                dataset_x = meta_data['x'].values
                dataset_y = meta_data['label'].values
                self.dataset_x_len = len(dataset_x)
                self.dataset_y_len = len(dataset_y)
        
    def __getitem__(self, index):
        if self.dataset_x is None:
            for file in os.listdir(self.file_path):
                if file.endswith(".csv"):
                    meta_data = pandas.read_csv(self.file_path + '/' + file)
                    self.dataset_x = meta_data['x'].values
        
        if self.dataset_y is None:
            for file in os.listdir(self.file_path):
                if file.endswith(".csv"):
                    meta_data = pandas.read_csv(self.file_path + '/' + file)
                    self.dataset_y = meta_data['label'].values

        image = Image.open(self.file_path + '/' + self.dataset_x[index])
        label = MHIST_LABELS[self.dataset_y[index]]
        if self.transform:
            image = self.transform(image)
        return image, label, index

    def __len__(self):
        assert self.dataset_x_len == self.dataset_y_len
        return self.dataset_x_len