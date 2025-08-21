import numpy as np
import cv2
import os
import json

def bilinear_interpolate(im, x, y):
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1

    if x0 < 0 or x1 >= im.shape[1] or y0 < 0 or y1 >= im.shape[0]:
        return 0

    Ia, Ib, Ic, Id = im[y0, x0], im[y1, x0], im[y0, x1], im[y1, x1]
    wa, wb = (x1 - x) * (y1 - y), (x1 - x) * (y - y0)
    wc, wd = (x - x0) * (y1 - y), (x - x0) * (y - y0)

    return np.clip(wa * Ia + wb * Ib + wc * Ic + wd * Id, 0, 255)

def rubberSheetNormalisation(img, mask, xPupil, yPupil, rPupil, xIris, yIris, rIris,
                              angle_samples=480, radius_samples=100, use_interpolation=True):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    angles = np.linspace(0, 2 * np.pi, angle_samples, endpoint=False)
    radii = np.linspace(0, 1, radius_samples)
    normalized = np.zeros((radius_samples, angle_samples), dtype=np.uint8)

    for i, theta in enumerate(angles):
        xp = xPupil + rPupil * np.cos(theta)
        yp = yPupil + rPupil * np.sin(theta)
        xi = xIris + rIris * np.cos(theta)
        yi = yIris + rIris * np.sin(theta)

        for j, r in enumerate(radii):
            x = (1 - r) * xp + r * xi
            y = (1 - r) * yp + r * yi

            if 0 <= int(y) < img.shape[0] and 0 <= int(x) < img.shape[1]:
                if mask[int(y), int(x)] == 2:
                    if use_interpolation:
                        normalized[j, i] = bilinear_interpolate(img, x, y)
                    else:
                        normalized[j, i] = img[int(round(y)), int(round(x))]
                else:
                    normalized[j, i] = 0
            else:
                normalized[j, i] = 0
    return normalized

def main():
    # Use current directory as base path
    project_root = os.getcwd()
    json_path = os.path.join(project_root, 'iris_localization_results_all.json')

    if not os.path.exists(json_path):
        print(f"❌ 找不到 JSON 檔案: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # ✅ Sort by image name for ascending order
    data = sorted(data, key=lambda x: x['image_name'])

    total = len(data)
    processed = 0

    for idx, item in enumerate(data):
        dataset = item.get('dataset', 'unknown')
        img_path = item['image_name']
        mask_path = item['mask_path']
        xPupil = item['center_x']
        yPupil = item['center_y']
        rPupil = item['pupil_radius']
        rIris = item['iris_radius']
        xIris, yIris = xPupil, yPupil  # Assume same center

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 找不到圖片: {img_path}")
            continue

        if not os.path.exists(mask_path):
            print(f"❌ 找不到 mask: {mask_path}")
            continue

        mask = np.load(mask_path)

        # Normalize
        normalized = rubberSheetNormalisation(img, mask, xPupil, yPupil, rPupil, xIris, yIris, rIris)

        # Extract path info
        parts = os.path.normpath(img_path).split(os.sep)
        try:
            subject_id = parts[-3]
            eye_side = parts[-2]
            image_name = os.path.splitext(parts[-1])[0]
        except IndexError:
            print(f"⚠️ 圖片路徑格式錯誤: {img_path}")
            continue

        gaze_label = "Gaze" if "Ganzin" in dataset else dataset.replace("CASIA_Iris_", "")
        output_dir = os.path.join(project_root, 'normalized_images', dataset)
        os.makedirs(output_dir, exist_ok=True)

        new_filename = f"{gaze_label}_{subject_id}_{eye_side}_{image_name}_normalized.png"
        save_path = os.path.join(output_dir, new_filename)

        # Save image
        cv2.imwrite(save_path, normalized)
        processed += 1
        print(f"✅ ({processed}/{total}) 已儲存: {save_path}")

    print(f"\n🎉 全部完成：成功處理 {processed} / {total} 筆資料")


if __name__ == "__main__":
    main()
