
import numpy as np
import clip
import torch
from tqdm import tqdm
from torch.functional import F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from Pcam.src.model.CustomCLIP import CustomCLIP
from dataset.Pcam import Pcam
from dataset.MHIST import MHIST
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
import sys
sys.path.insert(0, '../AI-Pathology')


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
    mhist = "dataset/DATA/MHIST/"
    DATA_DIR = "dataset/DATA/pcamv1/"
    test_mhist = mhist + 'test'
    train_path = DATA_DIR + 'camelyonpatch_level_2_split_train'
    valid_path = DATA_DIR + 'camelyonpatch_level_2_split_valid'
    test_path = DATA_DIR + 'camelyonpatch_level_2_split_test'

    train_dataset = Pcam(path=train_path, transform=preprocess)
    test_dataset = Pcam(path=test_path, transform=preprocess)
    val_dataset = Pcam(path=valid_path, transform=preprocess)
    pcam_class = ["tumor", "normal"]
    # init config

    # define model
    sample_image = train_dataset[0][0].unsqueeze(0).to(device)
    image_features = openai_clip.encode_image(sample_image)
    in_features = image_features.size()[1]

    percent = float(sys.argv[1])

    softmax = torch.nn.Softmax(dim=1)

    config_0 = config(backbone=backbone, CLIP=openai_clip, seed=None, percent=percent,
                      alpha=0, device=device, train_dataset=train_dataset, test_dataset=test_dataset)
    config_1 = config(backbone=backbone, CLIP=openai_clip, seed=None, percent=percent,
                      alpha=0.2, device=device, train_dataset=train_dataset, test_dataset=test_dataset)
    config_2 = config(backbone=backbone, CLIP=openai_clip, seed=None, percent=percent,
                      alpha=0.4, device=device, train_dataset=train_dataset, test_dataset=test_dataset)
    config_3 = config(backbone=backbone, CLIP=openai_clip, seed=None, percent=percent,
                      alpha=0.6, device=device, train_dataset=train_dataset, test_dataset=test_dataset)
    config_4 = config(backbone=backbone, CLIP=openai_clip, seed=None, percent=percent,
                      alpha=0.8, device=device, train_dataset=train_dataset, test_dataset=test_dataset)
    config_5 = config(backbone=backbone, CLIP=openai_clip, seed=None, percent=percent,
                      alpha=0.1, device=device, train_dataset=train_dataset, test_dataset=test_dataset)

    CLIP_RFC_0 = CustomCLIP(config_0, in_features=in_features).to(device)
    CLIP_RFC_1 = CustomCLIP(config_1, in_features=in_features).to(device)
    CLIP_RFC_1.load_state_dict(torch.load(
        "Pcam/src/model/checkpoints/CustomCLIP:{percent}_{alpha}_{seed}.pt".format(percent=percent, alpha=0.2, seed=1)))

    CLIP_RFC_2 = CustomCLIP(config_2, in_features=in_features).to(device)
    CLIP_RFC_2.load_state_dict(torch.load(
        "Pcam/src/model/checkpoints/CustomCLIP:{percent}_{alpha}_{seed}.pt".format(percent=percent, alpha=0.4, seed=1)))

    CLIP_RFC_3 = CustomCLIP(config_3, in_features=in_features).to(device)
    CLIP_RFC_3.load_state_dict(torch.load(
        "Pcam/src/model/checkpoints/CustomCLIP:{percent}_{alpha}_{seed}.pt".format(percent=percent, alpha=0.6, seed=1)))

    CLIP_RFC_4 = CustomCLIP(config_4, in_features=in_features).to(device)
    CLIP_RFC_4.load_state_dict(torch.load(
        "Pcam/src/model/checkpoints/CustomCLIP:{percent}_{alpha}_{seed}.pt".format(percent=percent, alpha=0.8, seed=1)))

    CLIP_RFC_5 = CustomCLIP(config_5, in_features=in_features).to(device)
    CLIP_RFC_5.load_state_dict(torch.load(
        "Pcam/src/model/checkpoints/CustomCLIP:{percent}_{alpha}_{seed}.pt".format(percent=percent, alpha=1.0, seed=1)))

    CLIP_RFC_GROUP = [CLIP_RFC_0, CLIP_RFC_1,
                      CLIP_RFC_2, CLIP_RFC_3, CLIP_RFC_4, CLIP_RFC_5]
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    alpha = 0
    for CLIP_RFC in CLIP_RFC_GROUP:
        prediction_list = []
        label_list = []

        for image, label, index in tqdm(DataLoader(test_dataset)):
            loss, logit = CLIP_RFC.forward(image.to(device), label.to(device))
            prediction = softmax(logit)
            prediction_list.append(prediction.cpu().detach().numpy())
            label_list.append(label.cpu().detach().numpy())

        prediction_list = np.concatenate(prediction_list)
        label_list = np.concatenate(label_list)

        fpr, tpr, _ = roc_curve(label_list, prediction_list[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3,
                 label='ROC fold %0.1f (AUC = %0.2f)' % (alpha, roc_auc))
        alpha += 0.2
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)

plt.xlabel('False Positive Rate', fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('True Positive Rate', fontsize=22)
plt.title('ROC', fontsize=26)
plt.legend(loc="lower right", fontsize=10)
plt.text(0.32, 0.7, 'More accurate ', fontsize=22)
# plt.text(0.63,0.4,'Less accurate area',fontsize = 22)
plt.savefig('./ROC_C.png', bbox_inches='tight', dpi=800)
plt.show()
