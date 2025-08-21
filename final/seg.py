# pip install torch torchvision numpy pillow matplotlib
# git clone https://github.com/AayushKrChaudhary/RITnet.git C:\Users\user\Documents\CV\final\RITnet

import os
import torch
from RITnet.densenet import DenseNet2D
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import sys

def main():
    project_root = r'C:\Users\user\Documents\CV\final'  
    sys.path.append(os.path.join(project_root, 'RITnet'))

    model_path = os.path.join(project_root, 'best_model.pkl')
    image_root = os.path.join(project_root, 'CASIA-Iris-Thousand')
    # image_root = os.path.join(project_root, 'CASIA-Iris-Lamp')
    # image_root = os.path.join(project_root, 'Ganzin-J7EF-Gaze')
    output_path = os.path.join(project_root, 'CASIA-Iris-Thousand_labels')
    # output_path = os.path.join(project_root, 'CASIA-Iris-Lamp_labels')
    # output_path = os.path.join(project_root, 'Ganzin-J7EF-Gaze_labels')

    os.makedirs(output_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DenseNet2D(dropout=True, prob=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    i = 0
    for num in range(0, 1000): # CASIA-Iris-Thousand
    # for num in range(0, 412): # CASIA-Iris-Lamp
    # for num in range(1, 11): # Ganzin-J7EF-Gaze
        num_str = f"{num:03d}"
        for eye in ['R', 'L']:
            for r_idx in range(0, 10):
            # for r_idx in range(1, 21):  # CASIA-Iris-Lamp

            # for a in range(1, 6): Ganzin-J7EF-Gaze
            #     for b in range(1, 11):
                r_str = f"{eye}{r_idx:02d}"
                image_path = os.path.join(image_root, f"{num_str}/{eye}/S5{num_str}{r_str}.jpg") # CASIA-Iris-Thousand
                # image_path = os.path.join(image_root, f"{num_str}/{eye}/S2{num_str}{r_str}.jpg") # CASIA-Iris-Lamp
                # image_path = os.path.join(image_root, f"{num_str}/{eye}/view_{a}_{b}.png") # Ganzin-J7EF-Gaze
                if not os.path.exists(image_path):
                    print(f"File not found: {image_path}")
                    continue
                image = Image.open(image_path).convert('L')
                original_size = image.size

                base_height = 400
                w_resize = int(image.size[0] * base_height / image.size[1])
                resized_image = image.resize((w_resize, base_height), Image.Resampling.LANCZOS)

                delta_w = 640 - w_resize
                pad_left = delta_w // 2
                pad_right = delta_w - pad_left
                pad_top = 0
                pad_bottom = 0

                img_np = np.array(resized_image)
                img_padded = np.pad(
                    img_np,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='reflect'
                )

                image_final = Image.fromarray(img_padded)

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                input_tensor = transform(image_final).unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)[0]
                    prediction = torch.argmax(output, dim=0).cpu().numpy()

                prediction_cropped = prediction[:, pad_left:640 - pad_right]
                prediction_pil = Image.fromarray(prediction_cropped.astype(np.uint8), mode='L')
                prediction_resized_back = prediction_pil.resize(original_size, Image.NEAREST)
                prediction_final = np.array(prediction_resized_back)

                save_dir = os.path.join(output_path, num_str, eye)
                os.makedirs(save_dir, exist_ok=True)

                label_filename = f"S5{num_str}{eye}{r_idx:02d}_labels.npy" # CASIA-Iris-Thousand
                # label_filename = f"S2{num_str}{eye}{r_idx:02d}_labels.npy" # CASIA-Iris-Lamp
                # label_filename = f"view_{a}_{b}_labels.npy" # Ganzin-J7EF-Gaze
                np.save(os.path.join(output_path, label_filename), prediction_final)
                print(f"Saved {label_filename}")

                if i < 2:
                    i += 1
                    plt.figure(figsize=(10, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(image, cmap='gray')
                    plt.title('Original Image')
                    plt.axis('off')

                    plt.subplot(1, 2, 2)
                    plt.imshow(prediction_final, cmap='gray')
                    plt.title('Segmentation Result')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.show()

if __name__ == '__main__':
    main()
