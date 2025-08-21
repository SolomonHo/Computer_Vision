# seg.py
# Undergo area segmentation for the eyes, to determine area of irises
# pip install torch torchvision numpy pillow matplotlib
# git clone https://github.com/AayushKrChaudhary/RITnet.git C:\Users\user\Documents\CV\final\RITnet

import os
import torch
from RITnet.densenet import DenseNet2D
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def process_dataset(model, device, dataset_name, image_root, output_path, num_range, eye_list, index_range_func, image_path_func, label_name_func):
    print(f"\nProcessing: {dataset_name}")
    os.makedirs(output_path, exist_ok=True)
    show_count = 0

    for num in num_range:
        num_str = f"{num:03d}"
        for eye in eye_list:
            for idx in index_range_func():
                image_path = image_path_func(num_str, eye, idx)
                if not os.path.exists(image_path):
                    print(f"[{dataset_name}] File not found: {image_path}")
                    continue

                image = Image.open(image_path).convert('L')
                original_size = image.size

                # Resize and pad
                base_height = 400
                w_resize = int(image.size[0] * base_height / image.size[1])
                resized_image = image.resize((w_resize, base_height), Image.Resampling.LANCZOS)
                delta_w = 640 - w_resize
                pad_left = delta_w // 2
                pad_right = delta_w - pad_left

                img_np = np.array(resized_image)
                img_padded = np.pad(img_np, ((0, 0), (pad_left, pad_right)), mode='reflect')
                image_final = Image.fromarray(img_padded)

                # Preprocess
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                input_tensor = transform(image_final).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    output = model(input_tensor)[0]
                    prediction = torch.argmax(output, dim=0).cpu().numpy()

                prediction_cropped = prediction[:, pad_left:640 - pad_right]
                prediction_pil = Image.fromarray(prediction_cropped.astype(np.uint8), mode='L')
                prediction_resized_back = prediction_pil.resize(original_size, Image.NEAREST)
                prediction_final = np.array(prediction_resized_back)

                # Save
                save_dir = os.path.join(output_path, num_str, eye)
                os.makedirs(save_dir, exist_ok=True)
                label_filename = label_name_func(num_str, eye, idx)
                np.save(os.path.join(save_dir, label_filename), prediction_final)
                print(f"[{dataset_name}] Saved {label_filename}")

                # # Show visualization (only first 2)
                # if show_count < 2:
                #     show_count += 1
                #     plt.figure(figsize=(10, 4))
                #     plt.subplot(1, 2, 1)
                #     plt.imshow(image, cmap='gray')
                #     plt.title('Original Image')
                #     plt.axis('off')

                #     plt.subplot(1, 2, 2)
                #     plt.imshow(prediction_final, cmap='gray')
                #     plt.title('Segmentation Result')
                #     plt.axis('off')

                #     plt.tight_layout()
                #     plt.show()

def main():
    home_dir = os.path.expanduser('~')
    project_root = os.getcwd()
    model_path = os.path.join(project_root, 'RITnet/best_model.pkl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DenseNet2D(dropout=True, prob=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- CASIA-Iris-Thousand ---
    process_dataset(
        model=model,
        device=device,
        dataset_name="CASIA-Iris-Thousand",
        image_root=os.path.join(project_root, 'dataset/CASIA-Iris-Thousand'),
        output_path=os.path.join(project_root, 'labels/CASIA-Iris-Thousand_labels'),
        num_range=range(0, 1000),
        eye_list=['L', 'R'],
        index_range_func=lambda: range(0, 10),
        image_path_func=lambda num, eye, r: os.path.join(project_root, 'dataset/CASIA-Iris-Thousand', f"{num}/{eye}/S5{num}{eye}{r:02d}.jpg"),
        label_name_func=lambda num, eye, r: f"S5{num}{eye}{r:02d}_labels.npy"
    )

    # --- CASIA-Iris-Lamp ---
    process_dataset(
        model=model,
        device=device,
        dataset_name="CASIA-Iris-Lamp",
        image_root=os.path.join(project_root, 'dataset/CASIA-Iris-Lamp'),
        output_path=os.path.join(project_root, 'labels/CASIA-Iris-Lamp_labels'),
        num_range=range(0, 412),
        eye_list=['L', 'R'],
        index_range_func=lambda: range(1, 21),
        image_path_func=lambda num, eye, r: os.path.join(project_root, 'dataset/CASIA-Iris-Lamp', f"{num}/{eye}/S2{num}{eye}{r:02d}.jpg"),
        label_name_func=lambda num, eye, r: f"S2{num}{eye}{r:02d}_labels.npy"
    )

    # --- Ganzin-J7EF-Gaze ---
    process_dataset(
        model=model,
        device=device,
        dataset_name="Ganzin-J7EF-Gaze",
        image_root=os.path.join(project_root, 'dataset/Ganzin-J7EF-Gaze'),
        output_path=os.path.join(project_root, 'labels/Ganzin-J7EF-Gaze_labels'),
        num_range=range(1, 11),
        eye_list=['L', 'R'],
        index_range_func=lambda: [(a, b) for a in range(1, 6) for b in range(1, 11)],
        image_path_func=lambda num, eye, ab: os.path.join(project_root, 'dataset/Ganzin-J7EF-Gaze', f"{num}/{eye}/view_{ab[0]}_{ab[1]}.png"),
        label_name_func=lambda num, eye, ab: f"view_{ab[0]}_{ab[1]}_labels.npy"
    )

if __name__ == '__main__':
    main()

