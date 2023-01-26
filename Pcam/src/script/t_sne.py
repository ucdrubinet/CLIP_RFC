from os.path import exists
import sys
sys.path.insert(0, '../AI-Pathology')
import time
import clip
import torch
import wandb
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from Pcam.src.model.CustomCLIP import CustomCLIP
from dataset.Pcam import Pcam
from dataset.MHIST import MHIST
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import random
from torch.utils.data import DataLoader, SubsetRandomSampler

import json

def cal_train_index(train_dataset, percent_train):
        """
        calculate the distrubution of the training set as of percent of training set, and ratio of each class
        now the ratio is 1:1, which is balanced
        """
        random.seed(1)

        # check if the distribution file exists
        if exists("Pcam/src/model/distribution.json"):
            print("Distribution file exists")
            with open("Pcam/src/model/distribution.json", "r") as f:
                dict_from_json = json.load(f)
                index_class = {0: dict_from_json["0"], 1: dict_from_json["1"]}
        
        # if not, calculate the distribution
        else:
            index_class = {0: [], 1: []}
            # get the index of each class
            for i in tqdm(range(train_dataset.__len__())):
                label = train_dataset.__getitem__(i)[1]
                index_class[label].append(i)
            with open("Pcam/src/model/distribution.json", "w") as f:
                json.dump(index_class, f)

        # total number of samples
        num_samples = int(train_dataset.__len__() * percent_train)

        # percent of samples from each class
        num_class_0 = int(num_samples * 0.5)
        num_class_1 = num_samples - num_class_0

        # randomly sample the index
        train_index_class_0 = random.sample(index_class[0], num_class_0)
        train_index_class_1 = random.sample(index_class[1], num_class_1)

        # combine the index
        train_index = train_index_class_0 + train_index_class_1
        print("Number of training samples: ", len(train_index))
        return train_index

class config():
    def __init__(self, backbone, CLIP, device, seed, percent, alpha, train_dataset, test_dataset):
        """
        config file for running CLIP + Residual Feature Connection
        @param device: CUDA or CPU
        @param backbone: Neural Network that CLIP trained on
        @param loss_fn: loss function
        @param scaler: scaler for mixed-percision training
        @param epochs: the number of epochs
        @param lr: learning rate
        @param alpha: percent feature from Residual Feature Connection Layer
        @param in_features: the input feature size
        @param percent: percent of total training set
        @param reduction: the reduction factor
        """
        self.device = device
        self.backbone = backbone
        self.CLIP = CLIP.float()
        self.scaler = GradScaler()
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # TODO adjust following hyperparameter to run different experiment
        self.lr = 1e-4
        self.weight_decay = 5e-3
        self.batch_size = 32
        self.epochs = 5
        self.alpha = alpha
        self.percent_training_set = percent
        self.seed = seed

        # initialize dataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.valid_dataset = None

if __name__ == "__main__":
    # init CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = "RN50"
    openai_clip, preprocess = clip.load(backbone, device)

    # define dataset
    DATA_DIR = "dataset/DATA/pcamv1/"
    train_path = DATA_DIR + 'camelyonpatch_level_2_split_train'
    valid_path = DATA_DIR + 'camelyonpatch_level_2_split_valid'
    test_path = DATA_DIR + 'camelyonpatch_level_2_split_test'

    train_dataset = Pcam(path=train_path, transform=preprocess)
    test_dataset = Pcam(path=test_path, transform=preprocess)
    val_dataset = Pcam(path=valid_path, transform=preprocess)
    pcam_class = ["Normal", "Tumor"]

    # init config
    percent = 0.01
    alpha = float(sys.argv[1])
    # define model
    sample_image = train_dataset[0][0].unsqueeze(0).to(device)
    image_features = openai_clip.encode_image(sample_image)
    in_features = image_features.size()[1]

    config = config(backbone=backbone, CLIP=openai_clip, seed=None, percent=percent,
                    alpha=alpha, device=device, train_dataset=train_dataset, test_dataset=test_dataset)

    CLIP_RFC = CustomCLIP(config=config, in_features=in_features).to(device)

    # load pretrained model
    if exists("Pcam/src/model/checkpoints/CustomCLIP:{percent}_{alpha}_{seed}.pt".format(percent=percent, alpha=alpha, seed=1)):
        print("Pretrained model exists")
        CLIP_RFC.load_state_dict(torch.load("Pcam/src/model/checkpoints/CustomCLIP:{percent}_{alpha}_{seed}.pt".format(percent=percent, alpha=alpha, seed=1)))

    X = []
    label_list = []
    train_index = cal_train_index(train_dataset=train_dataset, percent_train=percent)
    # init dataloader
    for image, label, index in tqdm(DataLoader(train_dataset, sampler=SubsetRandomSampler(train_index))):
        image = image.to(device)

        # get CLIP embedding
        image_features = CLIP_RFC.CLIP.encode_image(image)
        CLIP_embedding = CLIP_RFC.fc(image_features)


        # append embedding to X
        X.append(CLIP_embedding[0].tolist())
        label_list.append(pcam_class[int(label.cpu().numpy()[0])])
    
    # train t-SNE
    X = np.array(X)
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(X)

    df_subset = pd.DataFrame()
    df_subset['x_embedding'] = tsne_results[:,0]
    df_subset['y_embedding'] = tsne_results[:,1]
    df_subset['Class'] = label_list

    plt.figure(figsize=(16,10))
    sns.set(font_scale=1.7)
    sns.scatterplot(
        x="x_embedding", y="y_embedding",
        hue="Class",
        palette=sns.color_palette("hls", 2),
        data=df_subset,
        legend="full",
        alpha=0.3,
        s=100
    ).set(title="t-SNE visualization training={percent}%, alpha={alpha}".format(percent=str(percent*100), alpha=str(alpha)), xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
    plt.savefig('./Pcam/src/figure/t_sne_Pcam_0.1_%0.1f.png' % (alpha), dpi=800)