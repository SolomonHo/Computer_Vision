# dataset.py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class PairDatasetFromCSV(Dataset):
    def __init__(self, csv_file, transform=None):
        df = pd.read_csv(csv_file)
        self.pairs = list(zip(df['img1'], df['img2'], df['label']))
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),  # Convert to 1 channel
            transforms.Resize((100, 480)),  # [H, W]
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)

def get_pair_loader(csv_file, batch_size=16, shuffle=False):
    dataset = PairDatasetFromCSV(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
