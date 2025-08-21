import os
import numpy as np
import cv2
import json

project_root = os.getcwd()
dataset_root = os.path.join(project_root, 'dataset')
label_root = os.path.join(project_root, 'labels')

valid_datasets = ['CASIA-Iris-Lamp', 'CASIA-Iris-Thousand', 'Ganzin-J7EF-Gaze']
all_results = []

for dataset_name in valid_datasets:
    print(f"🔍 處理資料集: {dataset_name}")
    dataset_path = os.path.join(dataset_root, dataset_name)
    label_dataset_name = dataset_name + '_labels'
    label_path = os.path.join(label_root, label_dataset_name)

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if not file.lower().endswith(('.jpg', '.png')):
                continue

            img_path = os.path.join(root, file)

            # Build relative path and mask path
            relative_path = os.path.relpath(img_path, dataset_path)
            img_base, _ = os.path.splitext(relative_path)
            mask_filename = os.path.basename(img_base) + '_labels.npy'
            mask_subdir = os.path.dirname(relative_path)
            full_mask_path = os.path.join(label_path, mask_subdir, mask_filename)

            if not os.path.exists(full_mask_path):
                continue  # Skip images without masks

            # Load image and mask
            img = cv2.imread(img_path)
            mask = np.load(full_mask_path)

            pupil_mask = (mask == 3).astype(np.uint8)
            iris_mask = (mask == 2).astype(np.uint8)

            contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                pupil_contour = max(contours, key=cv2.contourArea)
                (pupil_x, pupil_y), pupil_radius = cv2.minEnclosingCircle(pupil_contour)
                pupil_center = (int(pupil_x), int(pupil_y))
                pupil_radius = int(pupil_radius)
            else:
                continue

            contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                iris_contour = max(contours, key=cv2.contourArea)
                (_, _), iris_radius = cv2.minEnclosingCircle(iris_contour)
                iris_radius = int(iris_radius)
            else:
                continue

            all_results.append({
                'dataset': dataset_name.replace('-', '_'),
                'image_name': img_path,
                'mask_path': full_mask_path,
                'center_x': pupil_center[0],
                'center_y': pupil_center[1],
                'pupil_radius': pupil_radius,
                'iris_radius': iris_radius
            })

output_json = os.path.join(project_root, 'iris_localization_results_all.json')
with open(output_json, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"\n✅ 處理完成，共 {len(all_results)} 筆資料已輸出至 iris_localization_results_all.json")
