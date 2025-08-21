import torch
from torch.nn.functional import pairwise_distance
import numpy as np
from dataset import get_pair_loader  # your DataLoader for pairs
from SiameseNN import SiameseNetwork
from SiameseNN_bestthou import SiameseNetwork_thou
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def tune_threshold(model, data_loader, device):
    model.eval()
    distances = []
    labels = []

    with torch.no_grad():
        for img1, img2, label in data_loader:
            img1, img2 = img1.to(device), img2.to(device)
            out1, out2 = model(img1, img2)
            dist = pairwise_distance(out1, out2)
            distances.append(dist.item())
            labels.append(label.item())

    distances = np.array(distances)
    labels = np.array(labels)

    best_acc = 0
    best_thresh = 0

    thresholds = np.linspace(distances.min(), distances.max(), 100)
    for thresh in thresholds:
        preds = (distances >= thresh).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    fpr, tpr, thresholds = roc_curve(labels, distances, pos_label=1)

    # Filter out non-finite threshold values
    finite_mask = np.isfinite(thresholds)
    fpr = fpr[finite_mask]
    tpr = tpr[finite_mask]
    thresholds = thresholds[finite_mask]

    # Compute Youden's J
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_thresh = thresholds[best_idx]

    # Use threshold to predict labels
    preds = (distances >= best_thresh).astype(int)
    best_acc = (preds == labels).mean()
    print(f"Thresholds range: {thresholds.min():.4f} to {thresholds.max():.4f}")
    print(f"Selected threshold index: {best_idx}, value: {best_thresh}")

    print(f"Best threshold (Youden's J): {best_thresh:.4f}, Accuracy: {best_acc:.4f}")
    with open('best_threshold.txt', 'w') as f:
        f.write(str(best_thresh))
    print(f"Best threshold: {best_thresh:.4f}, Accuracy: {best_acc:.4f}")
    with open('best_threshold.txt', 'w') as f:
        f.write(str(best_thresh))
    print(f"Best threshold saved: {best_thresh}")
        # Plotting histogram of distances
    genuine_distances = distances[labels == 0]
    impostor_distances = distances[labels == 1]

    plt.figure(figsize=(8, 5))
    plt.hist(genuine_distances, bins=30, alpha=0.6, label='Genuine (Label 0)', color='green')
    plt.hist(impostor_distances, bins=30, alpha=0.6, label='Impostor (Label 1)', color='red')
    plt.axvline(best_thresh, color='blue', linestyle='--', label=f'Threshold = {best_thresh:.2f}')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title('Distance Distribution by Pair Type')
    plt.legend()
    plt.tight_layout()
    plt.savefig('distance_distribution.png')
    print("[✓] Bar chart saved as distance_distribution.png")
    plt.close()

    return best_thresh, best_acc
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_pairdir', type=str, default='test_pairs.csv')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Choose model based on checkpoint filename
    if "siamese_best_thou" in os.path.basename(args.checkpoint).lower():
        model = SiameseNetwork_thou().to(device)
        print("[✓] Using SiameseNetwork_thou")
    else:
        model = SiameseNetwork().to(device)
        print("[✓] Using SiameseNetwork")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    val_loader = get_pair_loader(args.val_pairdir, batch_size=1)

    tune_threshold(model, val_loader, device)
