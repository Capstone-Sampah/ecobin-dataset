import os

PATH_DATASET = r"/dataset"
print(os.listdir(PATH_DATASET))

test_annotations_path = os.path.join(PATH_DATASET, "test", "_annotations.csv")
if os.path.exists(test_annotations_path):
    os.remove(test_annotations_path)

train_annotations_path = os.path.join(PATH_DATASET, "train", "_annotations.csv")
if os.path.exists(train_annotations_path):
    os.remove(train_annotations_path)

valid_annotations_path = os.path.join(PATH_DATASET, "valid", "_annotations.csv")
if os.path.exists(valid_annotations_path):
    os.remove(valid_annotations_path)