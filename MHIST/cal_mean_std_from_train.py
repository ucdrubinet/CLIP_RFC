import torchvision
import torch
from torch import tensor
from dataset.MHIST import MHIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# reference: https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html#:~:text=mean%3A%20simply%20divide%20the%20sum,%2F%20count%20%2D%20total_mean%20**%202)
DATA_DIR = "/home/zhli/Current-Work/Pcam_Experiment/dataset/DATA/MHIST/"
train_path = DATA_DIR + 'train'

# Calculate mean and stds
psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

train_data = MHIST(train_path, transform=transforms.ToTensor())

# Calculate mean and stds
for images, labels, index in tqdm(DataLoader(train_data)):
    psum += images.sum(dim=[0, 2, 3])
    psum_sq += (images ** 2).sum(dim=[0, 2, 3])
    print(psum)

n = train_data.__len__() * 224 * 224
mean = psum / n
std = (psum_sq / n - mean ** 2).sqrt()

print(mean, std)