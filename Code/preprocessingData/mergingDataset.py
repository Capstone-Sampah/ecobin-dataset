import os
import shutil

CLASSES = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]
PATH_DATASET = r"C:\Users\Felicia Pangestu\Documents\BANGKIT\Capstone\dataset"
PATH_DATASET_MERGED = r"C:\Users\Felicia Pangestu\Documents\BANGKIT\Capstone"
MERGED_DATASET_DIR = os.path.join(PATH_DATASET_MERGED, 'mergedDataset')

os.makedirs(MERGED_DATASET_DIR, exist_ok=True)
for class_name in CLASSES:
    class_combined_dir = os.path.join(MERGED_DATASET_DIR, class_name)
    os.makedirs(class_combined_dir, exist_ok=True)

    class_count = {}

    for dataset_name in ['train', 'valid', 'test']:
        dataset_dir = os.path.join(PATH_DATASET, dataset_name)
        for filename in os.listdir(dataset_dir):
            if class_name in filename:
                class_count[class_name] = class_count.get(class_name, 0) + 1
                file_extension = filename.split('.')[-1]
                new_filename = f"{class_name}{class_count[class_name]}.{file_extension}"
                src = os.path.join(dataset_dir, filename)
                dst = os.path.join(class_combined_dir, new_filename)
                shutil.copyfile(src, dst)