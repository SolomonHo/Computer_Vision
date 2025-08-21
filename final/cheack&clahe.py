import os
import cv2
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from densenet import DenseNet2D

def main():
    project_root = r'C:\Users\user\Documents\CV\final'  
    sys.path.append(os.path.join(project_root, 'RITnet'))

    model_path = os.path.join(project_root, 'best_model.pkl')
    image_root = os.path.join(project_root, 'CASIA-Iris-Thousand')
    # image_root = os.path.join(project_root, 'CASIA-Iris-Lamp')
    # image_root = os.path.join(project_root, 'Ganzin-J7EF-Gaze')
    label_root = os.path.join(project_root, 'CASIA-Iris-Thousand_labels')
    # label_root = os.path.join(project_root, 'CASIA-Iris-Lamp_labels')
    # label_root = os.path.join(project_root, 'Ganzin-J7EF-Gaze_labels')

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    threshold3 = 1000  # Iris
    threshold2 = 1000  # Pupil

    os.makedirs(label_root, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DenseNet2D(dropout=True, prob=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


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
                label_path = os.path.join(label_root, f"{num_str}/{eye}/S5{num_str}{r_str}_labels.npy") # CASIA-Iris-Thousand
                # label_path = os.path.join(label_root, f"{num_str}/{eye}/S2{num_str}{r_str}_labels.npy") # CASIA-Iris-Lamp
                # label_path = os.path.join(label_root, f"{num_str}/{eye}/view_{a}_{b}_labels.npy") # Ganzin-J7EF-Gaze
                if not os.path.exists(label_path):
                    print(f"File not found: {label_path}")
                    continue

                label = np.load(label_path)
                class_3_pixel_count = np.sum(label == 3)
                class_2_pixel_count = np.sum(label == 2)

                if class_3_pixel_count >= threshold3 and class_2_pixel_count >= threshold2:
                    continue  
                print(f"🔧 Need reconstruction:S5{num_str}{r_str} (label=3:{class_3_pixel_count}, label=2:{class_2_pixel_count})")

                raw_img_path = os.path.join(image_root, num_str, eye, f"S5{num_str}{r_str}.jpg")
                if not os.path.exists(raw_img_path):
                    print(f"❌ Cannot find the image: {raw_img_path}")
                    continue

                gray_img = cv2.imread(raw_img_path, cv2.IMREAD_GRAYSCALE)
                if gray_img is None:
                    print(f"❌ Failed to read image: {raw_img_path}")
                    continue
                
                enhanced = clahe.apply(gray_img)
                denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
                image = Image.fromarray(denoised)
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
                input_tensor = transform(image_final).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)[0]
                    prediction = torch.argmax(output, dim=0).cpu().numpy()

                prediction_cropped = prediction[:, pad_left:640 - pad_right]
                prediction_pil = Image.fromarray(prediction_cropped.astype(np.uint8), mode='L')
                prediction_resized_back = prediction_pil.resize(original_size, Image.NEAREST)
                prediction_final = np.array(prediction_resized_back)

                os.makedirs(os.path.dirname(label_path), exist_ok=True)
                np.save(label_path, prediction_final)
                print(f"✅ Restored label: {label_path}")

                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(gray_img, cmap='gray')
                plt.title('Original')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(prediction_final, cmap='gray')
                plt.title('Segmentation Result')
                plt.axis('off')
                plt.tight_layout()
                plt.show()

if __name__ == '__main__':
    main()
