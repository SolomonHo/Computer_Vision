import argparse
import os
import torch
import pandas as pd
from torch.nn.functional import pairwise_distance
from sklearn.metrics import accuracy_score
from SiameseNN import SiameseNetwork
from SiameseNN_bestthou import SiameseNetwork_thou

from dataset import get_pair_loader
from utils import write_csv
from torchvision import transforms
from torch.utils.data import DataLoader
import re

def parse_ganzin_txt_line(p):
    parts = p.strip().split("/")
    sid = parts[2]
    eye = parts[3]
    view, index = parts[4].replace(".png", "").split("_")[1:]
    return f"normalized_images/Ganzin_J7EF_Gaze/Gaze_{sid}_{eye}_view_{view}_{index}_normalized.png"

def convert_path(p):
    """
    Converts original path like:
    dataset/CASIA-Iris-Thousand/000/L/S5000L03.jpg
    to:
    normalized_images/CASIA-Iris-Thousand/L/S5000L03_normalized.png
    """
    parts = p.strip().split("/")
    if "CASIA-Iris-Thousand" in parts[1]:
        dataset = "CASIA-Iris-Thousand"
    elif "CASIA-Iris-Lamp" in parts[1]:
        dataset = "CASIA-Iris-Lamp"
    else:
        raise ValueError(f"Unknown dataset in path: {p}")
    eye = parts[3]
    filename = parts[4].replace(".jpg", "_normalized.png")
    return f"normalized_images/{dataset}/{eye}/{filename}"


def normalize_score(distance, alpha=1.5, center=1.0):
    return torch.sigmoid(torch.tensor(alpha * (distance - center))).item()

def load_best_threshold(dataset):
    threshold_files = {
        "thousand": "best_threshold_thou.txt",
        "lamp": "best_threshold_lamp.txt",
        "gaze": "best_threshold_gaze.txt",
        "bonus": "best_threshold_bonus.txt"
    }
    path = threshold_files.get(dataset)
    if path and os.path.exists(path):
        return float(open(path).read().strip())
    return None

def dataset_to_pairfile(dataset):
    return {
        "thousand": "output/CASIA-Iris-Thousand_normalized.csv",
        "lamp": "output/CASIA-Iris-Lamp_normalized.csv",
        "gaze": "output/Ganzin-J7EF-Gaze_normalized.csv",
        "bonus": "output/Ganzin-J7EF-Gaze_bonus_normalized.csv"

    }.get(dataset, None)

def dataset_to_listfile(dataset):
    return {
        "thousand": "dataset/list_CASIA-Iris-Thousand.txt",
        "lamp": "dataset/list_CASIA-Iris-Lamp.txt",
        "gaze": "dataset/list_Ganzin-J7EF-Gaze.txt",
        "bonus": "dataset/list_Ganzin-J7EF-Gaze_bonus.txt"
    }.get(dataset, None)

def extract_id(filename, dataset=None):
    import re
    match = re.search(r'(S\d{4}[LR]\d{2})', filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract ID from: {filename}")



def parse_ganzin_txt(txt_path):
    """
    Converts entries like:
    dataset/Ganzin-J7EF-Gaze/001/L/view_3_2.png
    to:
    images/Ganzin/Gaze_001_L_view_3_2_normalized.png
    (with enforced view numbers)
    """
    pairs = []
    with open(txt_path, 'r') as f:
        for line in f:
            left, right = line.strip().split(", ")
            l_parts = left.split("/")
            r_parts = right.split("/")

            def convert(parts):
                sid = parts[2]  # e.g., '007'
                eye = parts[3]  # 'L' or 'R'
                view, index = parts[4].replace(".png", "").split("_")[1:]  # ['view', '3'] → get '3'
                return f"normalized_images/Ganzin_J7EF_Gaze/Gaze_{sid}_{eye}_view_{view}_{index}_normalized.png"


            pairs.append((convert(l_parts), convert(r_parts)))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["thousand", "lamp", "gaze", "bonus"])
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/siamese_best_thou.pth')
    parser.add_argument('--output_csv', type=str, default='output.csv')
    parser.add_argument('--threshold', type=float, default=1.0)
    args = parser.parse_args()

    pair_file = dataset_to_pairfile(args.dataset)
    list_file = dataset_to_listfile(args.dataset)
    if pair_file is None or list_file is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"🔍 Evaluating dataset: {args.dataset} using {pair_file}")

    # Load model
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
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # Load test data
    df = pd.read_csv(pair_file)

    temp_csv = "temp_pairs.csv"
    df.to_csv(temp_csv, index=False)
    test_loader = get_pair_loader(temp_csv, batch_size=1)

    results = []
    all_labels = []
    all_preds = []

    best_threshold = load_best_threshold(args.dataset)
    threshold = best_threshold if best_threshold is not None else args.threshold
    print(f"📐 Using threshold: {threshold}")

    for (img1, img2, label), (_, row) in zip(test_loader, df.iterrows()):
        img1, img2 = img1.to(device), img2.to(device)
        label = label.item()
        with torch.no_grad():
            out1, out2 = model(img1, img2)
            dist = pairwise_distance(out1, out2).item()
            score = normalize_score(dist, alpha=1.5, center=threshold)
            pred = 0 if score < 0.5 else 1

        results.append({
            "img1": row['img1'],
            "img2": row['img2'],
            "label": int(label),
            "distance": dist,
            "score": score,
            "prediction": pred
        })

        all_labels.append(int(label))
        all_preds.append(pred)

    # Save results
    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"✅ Saved raw scores to {args.output_csv}")

    acc = accuracy_score(all_labels, all_preds)
    print(f"🎯 Accuracy @ threshold {threshold:.2f}: {acc:.4f}")

    # === Post-process: Insert scores into original pair list ===
    score_map = {}
    for r in results:
        if args.dataset in ["gaze", "bonus"]:
            def ganzin_path_to_id(p):
                basename = os.path.basename(p)
                # Expected format: Gaze_007_L_view_3_2_normalized.png
                name = basename.replace('_normalized.png', '')
                return name  # e.g., Gaze_007_L_view_3_2

            # print(f"🧩 img1 path: {r['img1']}")

            id1 = ganzin_path_to_id(r['img1'])
            id2 = ganzin_path_to_id(r['img2'])
            key = ",".join(sorted([id1, id2]))
            score_map[key] = r['prediction']


        else:
            id1 = extract_id(r['img1'], args.dataset)
            id2 = extract_id(r['img2'], args.dataset)
            key = ",".join(sorted([id1, id2]))
            score_map[key] = r['score']


        

    with open(list_file, 'r') as f:
        lines = f.readlines()

    scored_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        path1, path2 = [x.strip() for x in line.split(',')]
        # Convert to normalized filename paths, then extract keys
        if args.dataset in ["gaze", "bonus"]:
            norm1 = parse_ganzin_txt_line(path1)
            norm2 = parse_ganzin_txt_line(path2)
        else:
            norm1 = convert_path(path1)
            norm2 = convert_path(path2)

        id1 = os.path.basename(norm1).replace('_normalized.png', '')
        id2 = os.path.basename(norm2).replace('_normalized.png', '')
        key = ",".join(sorted([id1, id2]))



        score = score_map.get(key)
        if score is not None:
            scored_lines.append(f"{path1}, {path2}, {score:.4f}")
        else:
            scored_lines.append(f"{path1}, {path2}, SCORE_NOT_FOUND")
            print(f"[⚠️] Score not found for: {key}")

    # Save final list with scores
    if args.dataset in ["thousand", "lamp", "gaze"]:
        output_dir = "../test/"
    elif args.dataset == "bonus":
        output_dir = "../bonus/"
    else:
        output_dir = "./"
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build full output file path
    out_name = os.path.join(output_dir, f"result_{args.dataset}.txt")
    
    with open(out_name, 'w') as f:
        f.write('\n'.join(scored_lines))
    print(f"📄 Final scored list saved to {out_name}")

if __name__ == "__main__":
    main()
