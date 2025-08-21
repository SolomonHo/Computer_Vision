import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
from PIL import Image

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Load pretrained ResNet18
        self.resnet = models.resnet101(pretrained=True)
        # Modify the first conv layer to accept 1-channel input instead of 3
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final fully connected layer (fc)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # outputs (batch, 512, 1, 1)
        
        # New fully connected layers for embedding
        self.fc = nn.Sequential(
            nn.Flatten(),           # flatten (512)
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 128),     # output embedding size 128
            nn.ReLU(),
            nn.Linear(128, 64)     # output embedding size 128
        )

    def forward_once(self, x):
        x = self.resnet(x)    # shape: (batch, 512, 1, 1)
        x = self.fc(x)        # shape: (batch, 128)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = (1 - label) * torch.pow(euclidean_distance, 2) + \
               label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()

class IrisPairDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)