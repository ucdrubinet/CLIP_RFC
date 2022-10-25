from torch.utils.data import Dataset
from torchvision import datasets
import h5py
import matplotlib as plt
import numpy as np
from PIL import Image


class Pcam(Dataset):
    def __init__(self, path, transform=None):
        self.file_path = path
        self.dataset_x = None
        self.dataset_y = None
        self.transform = transform
        print(self.file_path)

        # Going to read the X part of the dataset - it's a different file
        with h5py.File(self.file_path + '_x.h5', 'r') as file_x:
            print(self.file_path)
            self.dataset_x_len = len(file_x['x'])

        # Going to read the y part of the dataset - it's a different file
        with h5py.File(self.file_path + '_y.h5', 'r') as file_y:
            self.dataset_y_len = len(file_y['y'])

    def __getitem__(self, index):
        if self.dataset_x is None:
            self.dataset_x = h5py.File(self.file_path + '_x.h5', 'r')['x']
        if self.dataset_y is None:
            self.dataset_y = h5py.File(self.file_path + '_y.h5', 'r')['y']

        image = Image.fromarray(self.dataset_x[index])
        label = self.dataset_y[index].item()
        if self.transform:
            image = self.transform(image)
        return image, label, index

    def __len__(self):
        assert self.dataset_x_len == self.dataset_y_len
        return self.dataset_x_len

    def visualize(self, index):
        if self.dataset_x is None:
            self.dataset_x = h5py.File(self.file_path + '_x.h5', 'r')['x']
        if self.dataset_y is None:
            self.dataset_y = h5py.File(self.file_path + '_y.h5', 'r')['y']
        image = self.dataset_x[index]
        #         if self.transform:
        #             image = self.transform(image)
        #         print('Label: ', self.dataset_y[index].item())
        #         return self.dataset_y[index].item()
        plt.imshow(image)
        plt.show
