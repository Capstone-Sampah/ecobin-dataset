import os
import shutil
from sklearn.model_selection import train_test_split

PATH_DATASET_SPLITED = r"C:\Users\Felicia Pangestu\Documents\BANGKIT\Capstone\splitedDataset"
PATH_DATASET_MERGED = r"C:\Users\Felicia Pangestu\Documents\BANGKIT\Capstone\mergedDataset"

CLASSES = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

for class_name in CLASSES:
    class_dir = os.path.join(PATH_DATASET_MERGED, class_name)
    class_files = os.listdir(class_dir)

    # Split the filenames into train, validation, and test sets
    train_files, remaining_files = train_test_split(class_files, test_size=1.0 - TRAIN_SPLIT, random_state=28)
    validation_files, test_files = train_test_split(remaining_files, test_size=TEST_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT), random_state=28)

    # Create subdirectories within the 'combined' directory for train_ds, val_ds, and test_ds
    train_combined_dir = os.path.join(PATH_DATASET_SPLITED, 'train_ds', class_name)
    valid_combined_dir = os.path.join(PATH_DATASET_SPLITED, 'val_ds', class_name)
    test_combined_dir = os.path.join(PATH_DATASET_SPLITED, 'test_ds', class_name)

    # Clear existing directories if they exist
    if os.path.exists(train_combined_dir):
        shutil.rmtree(train_combined_dir, ignore_errors=True)
    if os.path.exists(valid_combined_dir):
        shutil.rmtree(valid_combined_dir, ignore_errors=True)
    if os.path.exists(test_combined_dir):
        shutil.rmtree(test_combined_dir, ignore_errors=True)

    # Recreate directories
    os.makedirs(train_combined_dir, exist_ok=True)
    os.makedirs(valid_combined_dir, exist_ok=True)
    os.makedirs(test_combined_dir, exist_ok=True)

    # Copy files to the corresponding subdirectories within the 'combined' directory
    for filename in train_files:
        src = os.path.join(class_dir, filename)
        dst = os.path.join(train_combined_dir, filename)
        shutil.copyfile(src, dst)

    for filename in validation_files:
        src = os.path.join(class_dir, filename)
        dst = os.path.join(valid_combined_dir, filename)
        shutil.copyfile(src, dst)

    for filename in test_files:
        src = os.path.join(class_dir, filename)
        dst = os.path.join(test_combined_dir, filename)
        shutil.copyfile(src, dst)