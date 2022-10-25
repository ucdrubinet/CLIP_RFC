import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.Pcam import Pcam
from tqdm import tqdm
import sklearn.metrics as metrics
import sys
import wandb

# define hyperparameters
max_epoch = 5
lr = 3e-4
loss_fn = torch.nn.CrossEntropyLoss()
DATA_DIR = "/home/zhli/Current-Work/Pcam_Experiment/dataset/DATA/pcamv1/"
device = "cuda" if torch.cuda.is_available() else "cpu"
wandb.init(project="Pcam")

def print_metrices(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    wandb.log({"Accuracy": accuracy, "Precision": precision,
              "Recall": recall, "F1": f1})


def train(train_data, test_data, model, epochs):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    wandb.watch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        running_score = 0.0
        running_loss = 0.0
        for images, labels, index in tqdm(DataLoader(train_data, batch_size=128)):
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

            val, index_ = torch.max(y_pred, axis=1)
            running_score += torch.sum(index_ == label.data).item()
            running_loss += loss.item()
            print("loss: ", loss)
            wandb.log({"Training loss - Step": loss})

        epoch_score = running_score/train_data.__len__()
        epoch_loss = running_loss/train_data.__len__()
        print("Training loss: {}, accuracy: {}".format(epoch_loss, epoch_score))
        wandb.log({"Training loss - Epoch": epoch_loss,
                  "Training accuracy": epoch_score})
    # recieve 
    # test
    with torch.no_grad():
        label_list = []
        preds = []
        for images, labels, index in tqdm(DataLoader(test_data)):
            images = images.to(device)
            label = labels.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(images)
            _, predicted = torch.max(y_pred.data, 1)
            label_list.append(label.to("cpu").numpy())
            preds.append(predicted.to("cpu").numpy())

        print_metrices(label_list, preds)

if __name__ == "__main__":
    train_path = DATA_DIR + 'camelyonpatch_level_2_split_train'
    valid_path = DATA_DIR + 'camelyonpatch_level_2_split_valid'
    test_path = DATA_DIR + 'camelyonpatch_level_2_split_test'

    preprocess_pretrain = transforms.Compose([
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                0.229, 0.224, 0.225])
    ])

    train_dataset = Pcam(path=train_path, transform=preprocess_pretrain)
    test_dataset = Pcam(path=test_path, transform=test_transform)
    val_dataset = Pcam(path=valid_path, transform=preprocess_pretrain)

    model_backbone = torchvision.models.resnet50().to(device)
    train(train_dataset, test_dataset, model_backbone, max_epoch)