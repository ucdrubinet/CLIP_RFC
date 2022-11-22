import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
import sklearn.metrics as metrics
from tqdm import tqdm
import sys

# add the directory to the path
sys.path.insert(0, '../Pcam_Experiment')

from dataset.Pcam import Pcam

# Load parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)
batch_size = 32
epochs = 10

# define the model
class CustomCLIP(nn.Module):
    def __init__(self, in_features, reduction=4):
        super(CustomCLIP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features // reduction, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, in_features // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features // reduction, in_features, bias=False),
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
# define the training function
def train(train_data, test_data, model, clip_model, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)
    ratio = 0.3
    for _ in range(epochs):
        model.train(True)
        running_score = 0.0
        running_loss = 0.0
        for images, labels, index in tqdm(DataLoader(train_data, batch_size=batch_size, shuffle=True)):
            image_input = images.to(device)
            text_input = torch.cat([clip.tokenize(
                "this is a photo of healthy lymph node tissue"), clip.tokenize("this is a photo of lymph node tumor tissue")]).to(device)

            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            print(image_features.shape)
            pred_image_features = model.forward(image_features)
            pred_image_features /= pred_image_features.norm(dim=-1, keepdim=True)

            consisted_image_features = ratio * pred_image_features + (1 - ratio) * image_features
            optimizer.zero_grad()

            similarity = (100.0 * consisted_image_features @ text_features.T)
            print(similarity)

            loss = loss_fn(similarity, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print("loss: ", loss)

        epoch_loss = running_loss/train_data.__len__()
        print("Training loss: {}".format(epoch_loss))

        
if __name__ == "__main__":
    # define dataset
    DATA_DIR = "dataset/DATA/pcamv1/"
    train_path = DATA_DIR + 'camelyonpatch_level_2_split_train'
    valid_path = DATA_DIR + 'camelyonpatch_level_2_split_valid'
    test_path = DATA_DIR + 'camelyonpatch_level_2_split_test'

    train_dataset = Pcam(path=train_path, transform=preprocess)
    test_dataset = Pcam(path=test_path, transform=preprocess)
    val_dataset = Pcam(path=valid_path, transform=preprocess)
    
    # define model
    sample_image = train_dataset[0][0].unsqueeze(0).to(device)
    image_features = model.encode_image(sample_image)

    fc_layer = CustomCLIP(image_features.shape[1]).to(device)
    train(train_dataset, test_dataset, fc_layer, model, epochs)