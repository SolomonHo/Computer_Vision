import os
import glob
import csv

default_label = 0  # Change if needed (e.g., 1 for all positive pairs)

def extract_user_id(filepath):
    """
    Extract user ID (e.g., '001', 'S5000') from the filename
    """
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    # Common pattern: Dataset_User_Eye_...normalized.png
    # Example: Lamp_001_L_S2001L06_normalized.png → '001'
    #          Thousand_S5000_L_S5000L00_normalized.png → 'S5000'
    for part in parts:
        if part.isdigit() or part.startswith('S'):
            return part
    return None  # fallback if unrecognized


def convert_path(path):
    path = path.strip()
    if 'Ganzin' in path:
        parts = path.split('/')
        dataset_folder = 'normalized_images/Ganzin_J7EF_Gaze'
        subject = parts[2]
        eye = parts[3]
        view_name = os.path.splitext(parts[4])[0]
        filename = f'Gaze_{subject}_{eye}_{view_name}_normalized.png'
        return os.path.join(dataset_folder, filename)

    elif 'Thousand' in path:
        parts = path.split('/')
        dataset_folder = 'normalized_images/CASIA_Iris_Thousand'
        subject = parts[2]
        eye = parts[3]
        base_name = os.path.splitext(parts[4])[0]
        filename = f'Thousand_{subject}_{eye}_{base_name}_normalized.png'
        return os.path.join(dataset_folder, filename)

    elif 'Lamp' in path:
        parts = path.split('/')
        dataset_folder = 'normalized_images/CASIA_Iris_Lamp'
        subject = parts[2]
        eye = parts[3]
        base_name = os.path.splitext(parts[4])[0]
        filename = f'Lamp_{subject}_{eye}_{base_name}_normalized.png'
        return os.path.join(dataset_folder, filename)

    else:
        raise ValueError(f"Unknown dataset path: {path}")

# Find all input text files that start with "list_"
# Search both test/ and bonus/ directories
input_files = glob.glob("../test/list_*.txt") + glob.glob("../bonus/list_*.txt")

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

for input_txt in input_files:
    dataset_id = os.path.basename(input_txt).replace("list_", "").replace(".txt", "")
    output_txt = os.path.join(output_dir, f"{dataset_id}_normalized.txt")
    output_csv = os.path.join(output_dir, f"{dataset_id}_normalized.csv")
    print(f"[INFO] Processing: {input_txt}")

    # Write normalized text
    with open(input_txt, 'r') as infile, open(output_txt, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                print(f"[WARNING] Skipping malformed line in {input_txt}: {line}")
                continue
            try:
                new_paths = [convert_path(p) for p in parts]
                outfile.write(', '.join(new_paths) + '\n')
            except Exception as e:
                print(f"[ERROR] {e} on line: {line}")

    # Convert to CSV with inferred labels
    with open(output_txt, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['img1', 'img2', 'label'])
        for line in infile:
            parts = line.strip().split(',')
            if len(parts) == 2:
                img1 = parts[0].strip()
                img2 = parts[1].strip()

                # Extract subject IDs from filenames
                def extract_subject_id(path):
                    base = os.path.basename(path)
                    if base.startswith("Gaze_"):
                        return base.split('_')[1]
                    elif base.startswith("Thousand_"):
                        return base.split('_')[1]
                    elif base.startswith("Lamp_"):
                       return base.split('_')[1]
                    else:
                        return None

                id1 = extract_subject_id(img1)
                id2 = extract_subject_id(img2)
                label = int(id1 != id2) if id1 and id2 else default_label

                writer.writerow([img1, img2, label])
            else:
                print(f"[WARNING] Skipping malformed CSV line in {output_txt}: {line}")

    print(f"[✓] Finished: {output_txt}, {output_csv}")
