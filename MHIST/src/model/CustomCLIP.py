import torch
import json
import wandb
import datetime as dt
from tqdm import tqdm
import clip
import torch.nn as nn
import sklearn.metrics as metrics
import random 
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.cuda.amp import autocast
from os.path import exists


def print_metrices(y_true, y_pred, y_score, y_class_score):
    """
    print the metrices
    @param y_true: the true label
    @param y_pred: the predicted label
    @param y_score: the predicted score
    """
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    AUC = metrics.roc_auc_score(y_true, y_score)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("AUC: ", AUC)

    wandb.log({"Test Accuracy": accuracy, "Test Precision": precision,
              "Test Recall": recall, "Test F1": f1, "Test AUC": AUC})
    
    # following https://docs.wandb.ai/guides/track/log/plots
    wandb.log({"roc": wandb.plot.roc_curve(
        y_true, y_class_score, labels=["hyperplastic polyp", "sessile serrated adenoma"])})

class CustomCLIP(nn.Module):
    def __init__(self, config, in_features, reduction=4):
        """
        @param config: config file for running CLIP + Residual Feature Connection
        @param in_features: the input feature size
        @param reduction: the reduction factor
        """
        super(CustomCLIP, self).__init__()
        
        # define the hyperparameters from config
        self.CLIP = config.CLIP
        self.scaler = config.scaler
        self.softmax = nn.Softmax(dim=1)
        self.epochs = config.epochs
        self.device = config.device
        self.loss_fn = config.loss_fn
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.batch_size = config.batch_size
        self.alpha = config.alpha
        self.percent_train = config.percent_training_set
        self.seed = config.seed

        # define the dataset from config
        self.train_dataset = config.train_dataset
        self.test_dataset = config.test_dataset

        # TODO add text input to match different dataset
        self.text_input = torch.cat([clip.tokenize(
            "this is a photo of hyperplastic polyp"), clip.tokenize("this is a photo of sessile serrated adenoma")]).to(self.device)

        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features // reduction, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, in_features // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features // reduction, in_features, bias=False),
            nn.ReLU(),
        )

    def cal_train_index(self):
        """
        calculate the distrubution of the training set as of percent of training set, and ratio of each class
        now the ratio is 1:1, which is balanced
        """
        random.seed(self.seed)

        # check if the distribution file exists
        if exists("MHIST/src/model/distribution.json"):
            print("Distribution file exists")
            with open("MHIST/src/model/distribution.json", "r") as f:
                dict_from_json = json.load(f)
                index_class = {0: dict_from_json["0"], 1: dict_from_json["1"]}
        
        # if not, calculate the distribution
        else:
            index_class = {0: [], 1: []}
            # get the index of each class
            for i in tqdm(range(self.train_dataset.__len__())):
                label = self.train_dataset.__getitem__(i)[1]
                index_class[label].append(i)
            with open("MHIST/src/model/distribution.json", "w") as f:
                json.dump(index_class, f)

        # total number of samples
        num_samples = int(self.train_dataset.__len__() * self.percent_train)

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

    def forward(self, image_input, label):
       # freeze the CLIP model, and use as a feature extractor
        image_features = self.CLIP.encode_image(image_input)
        text_features = self.CLIP.encode_text(self.text_input)
        
        # mixed-precision training with autocast,
        # reference: https://developer.nvidia.com/blog/video-mixed-precision-techniques-tensor-cores-deep-learning/;
        # https://github.com/openai/CLIP/issues/57
        with autocast():
            pred_image_features = self.fc(image_features)
            pred_image_features = pred_image_features / \
                pred_image_features.norm(dim=-1, keepdim=True)

            image_features = image_features / \
                image_features.norm(dim=-1, keepdim=True)

            text_features = text_features / \
                text_features.norm(dim=-1, keepdim=True)

            # TODO ask jeff about deciding logit_scale or temprature
            # reference: https://github.com/huggingface/transformers/issues/13430
            logit_scale = self.CLIP.logit_scale.exp() 
            logit = logit_scale * (self.alpha * pred_image_features + (1 - self.alpha) * image_features) @ text_features.t()
            loss = self.loss_fn(logit, label).to(self.device)
        return loss, logit

    def save(self):

        path = './MHIST/src/model/checkpoints/CustomCLIP:' + \
            str(self.percent_train) + '_' + \
            str(self.alpha) + '_' + str(self.seed) + '.pt'

        if not exists(path):
            torch.save(self.state_dict(), path)
            print("Model saved to {}".format(path))
            
        else:
            print("Model already exists")

    def train(self):
        optimizer = torch.optim.Adam(
            self.fc.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sampling_size = int(self.train_dataset.__len__() * self.percent_train)
        pred = []
        true = []
        train_index = self.cal_train_index()
        for _ in range(self.epochs):
            running_score = 0.0
            running_loss = 0.0
            for images, labels, index in tqdm(DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(train_index))):
                image_input = images.to(self.device)
                labels = labels.to(self.device)

                loss, logit = self.forward(image_input, label=labels)
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                val, index = torch.max(logit, 1)
                label = labels.data.cpu().numpy()
                pred.extend(index.data.cpu().numpy())
                true.extend(label)

                running_loss += loss.item()
                # wandb.log({"Training loss - Step": loss.item(),
                #            "Training accuracy - Step": metrics.accuracy_score(true, pred)})

            epoch_loss = running_loss/sampling_size
            # wandb.log({"Training loss - Epoch": epoch_loss,
            #         "Training accuracy": metrics.accuracy_score(true, pred)})

        print("Saving model...")
        self.save()
    
    def test(self):
        pred = []
        true = []
        score = [] # for roc_auc metric, only record prob of true label in binary classification
        class_score = [] # for roc metric, record both false and true label probability
        for images, labels, index in tqdm(DataLoader(self.test_dataset)):
            image_input = images.to(self.device)
            labels = labels.to(self.device)

            loss, logit = self.forward(image_input, label=labels)
            val, index_ = torch.max(logit, 1)
            label = labels.data.cpu().numpy()
            true.append(int(label))
            pred.append(int(index_.cpu().numpy()))

            # Calculate probability of each class, and match to the label of the class for AUCs
            softmax_score = self.softmax(logit)
            prob = softmax_score.cpu().detach().numpy()

            class_score.append(list(prob[0]))
            score.append(float(prob[0][1]))
            
            wandb.log({"Test loss - Step": loss.item(),
                       "Test accuracy - Step": metrics.accuracy_score(true, pred)})

        print_metrices(true, pred, score, class_score)