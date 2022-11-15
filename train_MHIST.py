import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.MHIST import MHIST
from dataset.Pcam import Pcam
from tqdm import tqdm
import sklearn.metrics as metrics
import wandb
import numpy as np
from PIL import Image
from typing import (Tuple)
import random
import time

class Random90Rotation:
    """
    randomly rotate the training images in 90 degree increments
    @param img: the image to rotate
    @return: the rotated image
    """
    def __init__(self, degrees: Tuple[int] = None) -> None:
        """
        Randomly rotate the image for training. Credits to Naofumi Tomita.
        Args:
            degrees: Degrees available for rotation.
        """
        self.degrees = (0, 90, 180, 270) if (degrees is None) else degrees

    def __call__(self, im: Image) -> Image:
        """
        Produces a randomly rotated image every time the instance is called.
        Args:
            im: The image to rotate.
        Returns:
            Randomly rotated image.
        """
        return im.rotate(angle=random.sample(population=self.degrees, k=1)[0])


# Hyperparameters
wandb.init(project="MHIST")
DATA_DIR = "/home/zhli/Current-Work/Pcam_Experiment/dataset/DATA/MHIST/"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Experiment parameters
max_epoch = 100
lr = 1e-3
batch_size = 32
rate_decay = 0.91
seed = 0
loss_fn = torch.nn.CrossEntropyLoss()
model_backbone = torchvision.models.resnet50().to(device)
argumen = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1,
                           saturation=0.1, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    Random90Rotation(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

wandb.config.update({
    "epochs": max_epoch,
    "lr": lr,
    "batch_size": batch_size,
    "scheduler": "ExponentialLR",
    "rate_decay": rate_decay,
    "Shuffle": True,
    "Augmentation": True,
    "Augmentation_type": argumen.__repr__(),
    "seed": seed,
    "Backbone": "ResNet50",
    "Pretrained": False,
})


def print_metrices(y_true, y_pred, y_score, record):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    AUC = metrics.roc_auc_score(y_true, y_score)

    record["accuracy"].append(accuracy)
    record["precision"].append(precision)
    record["recall"].append(recall)
    record["f1"].append(f1)
    record["AUC"].append(AUC)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("AUC: ", AUC)

    wandb.log({"Accuracy - Valid": accuracy, "Precision - Valid": precision,
              "Recall - Valid": recall, "F1 - Valid": f1, "AUC - Valid": AUC})


def evluate_AUC(record):
    AUC = record["AUC"]
    recall = record["recall"]
    f1 = record["f1"]
    accuracy = record["accuracy"]
    precision = record["precision"]

    sort_index = np.argsort(AUC)
    top_5_index = sort_index[(len(sort_index) - 5):]

    for i in top_5_index:
        wandb.log({"Top-5 AUC": AUC[i], "Accuracy": accuracy[i],
                   "Recall": recall[i], "F1": f1[i], "Precision": precision[i]})


def train(train_data, test_data, model, epochs):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2).to(device)
    softmax = nn.Softmax(dim=1)
    wandb.watch(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr)
    train_label = []
    train_pred = []
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=rate_decay)

    record = {"accuracy": [], "precision": [],
              "recall": [], "f1": [], "AUC": []}

    for _ in range(epochs):
        model.train(True)
        running_score = 0.0
        running_loss = 0.0
        label_list = []
        preds = []
        scores = []
        for images, labels, index in tqdm(DataLoader(train_data, batch_size=batch_size, shuffle=True)):
            images = images.to(device)
            label = labels.to(device)

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()

            y_pred = model.forward(images)
            loss = loss_fn(y_pred, label)

            loss.backward()
            optimizer.step()

            val, index_ = torch.max(y_pred.data, axis=1)
            running_score += torch.sum(index_ == label.data).item()
            running_loss += loss.item()
            train_label.extend(label.data.cpu().numpy())
            train_pred.extend(index_.cpu().numpy())
            print("loss: ", loss)
            wandb.log({"Training loss - Step": loss})

        scheduler.step()
        epoch_score = running_score/train_data.__len__()
        epoch_loss = running_loss/train_data.__len__()

        print("Training loss: {}, accuracy: {}".format(epoch_loss, epoch_score))

        """
        Evluating Model in every epoch
        """
        model.train(False)
        for images, labels, _ in tqdm(DataLoader(test_data)):
            images = images.to(device)
            label = labels.to(device)
            y_pred = model.forward(images)
            
            # Calculate the Accuracy, Precision, Recall, F1 from label and y_pred
            val, index_ = torch.max(y_pred.data, axis=1)
            label = label.data.cpu().numpy()
            label_list.append(int(label))
            preds.append(int(index_.cpu().numpy()))

            # Calculate probability of each class, and match to the label of the class for AUCs
            softmax_score = softmax(y_pred)
            prob = softmax_score.cpu().detach().numpy()
            scores.append(float(prob[0][1]))

        print_metrices(label_list, preds, scores, record)

        wandb.log({"Training loss - Epoch": epoch_loss,
                  "Training accuracy": epoch_score})

    # save the model
    torch.save(model.state_dict(), "MHIST_model.pth")

    # evluate the top-5 AUC, and its corresponding accuracy, recall, f1, precision
    evluate_AUC(record)


if __name__ == "__main__":

    train_path = DATA_DIR + 'train'
    test_path = DATA_DIR + 'test'

    train_data = MHIST(train_path, transform=argumen)

    test_data = MHIST(test_path, transform=transforms.Compose([
        transforms.ToTensor(),

        # mean and std of the dataset for normalization from train set
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
        ]))

    torch.manual_seed(seed=seed)
    train(train_data, test_data, model_backbone, max_epoch)