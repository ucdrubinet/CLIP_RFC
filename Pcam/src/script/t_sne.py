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

    test_dataset_m = MHIST(path=test_mhist, transform=preprocess)
    mhist_class = ["hyperplastic papilloma", "sessile serrated adenoma"]


    # init config

    # define model
    sample_image = train_dataset[0][0].unsqueeze(0).to(device)
    image_features = openai_clip.encode_image(sample_image)
    in_features = image_features.size()[1]

    config = config(backbone=backbone, CLIP=openai_clip, seed=None, percent=None,
                    alpha=1, device=device, train_dataset=train_dataset, test_dataset=test_dataset)

    CLIP_RFC = CustomCLIP(config=config, in_features=in_features).to(device)

    CLIP_RFC.load_state_dict(torch.load("Pcam/src/model/checkpoints/CustomCLIP:2022-12-17_21-35-50.pt"))

    # init wandb
    wandb.init(project='CustomCLIP_Validation')
    pretrained_embedding_table = wandb.Table(columns=['label', 'embedding'])
    # init dataloader
    for image, label, index in tqdm(DataLoader(train_dataset)):
        image = image.to(device)

        # get CLIP embedding
        image_features = CLIP_RFC.CLIP.encode_image(image)
        CLIP_embedding = CLIP_RFC.fc(image_features)

        pretrained_embedding_table.add_data(pcam_class[label], CLIP_embedding[0].tolist())

    print("Logging embedding table")
    wandb.log({'embedding_table': pretrained_embedding_table})
    wandb.finish()
