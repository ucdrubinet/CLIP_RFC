import sys
sys.path.insert(0, '../AI-Pathology')
from dataset.Pcam import Pcam
from Pcam.src.model.CustomCLIP import CustomCLIP
from torch.cuda.amp import GradScaler
import wandb
import torch
import clip


class config():
    def __init__(self, backbone, CLIP, device, train_dataset, test_dataset, val_dataset):
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
        self.alpha = 0.8
        self.percent_training_set = 0.001
        self.seed = 1

        # initialize dataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.valid_dataset = val_dataset

        # initialize weights and biases
        self.init_wandb()

    def init_wandb(self):
        wandb.init(project="CustomCLIP")

        wandb.config.update({
            "dataset": "Pcam",
            "seed": self.seed,
            "epoch": self.epochs,
            "alpha": self.alpha,
            "percent_data_training": str(self.percent_training_set*100) + "%",
            "batch_size": self.batch_size,
            "lr": self.lr,
            "backbone": "CLIP " + self.backbone + "Residual Feature Connection",
            "scaler": "GradScaler",
            "loss_fn": "CrossEntropyLoss"
        })


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

    # init config
    config = config(backbone=backbone, CLIP=openai_clip, device=device, train_dataset=train_dataset,
                    test_dataset=test_dataset, val_dataset=val_dataset)
    # define model
    sample_image = train_dataset[0][0].unsqueeze(0).to(device)
    image_features = openai_clip.encode_image(sample_image)
    in_features = image_features.size()[1]

    CLIP_RFC = CustomCLIP(config=config, in_features=in_features).to(device)

    # training on dataset
    # CLIP_RFC.train()
  
    # testing 
    CLIP_RFC.test()
