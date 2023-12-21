import os
from collections import defaultdict

def count_images_per_class_and_total(folder_path):
    class_counts = defaultdict(int)
    total_count = 0

    for class_name in CLASSES:
        class_folder_path = os.path.join(folder_path, class_name)
        count = len(os.listdir(class_folder_path))
        class_counts[class_name] = count
        total_count += count

    return class_counts, total_count

# Ganti 'path/to/folder' dengan jalur aktual menuju setiap folder (test_ds, valid_ds, train_ds)
test_folder_path = r'C:\Users\Felicia Pangestu\Documents\BANGKIT\Capstone\splitedDataset\test_ds'
valid_folder_path = r'C:\Users\Felicia Pangestu\Documents\BANGKIT\Capstone\splitedDataset\val_ds'
train_folder_path = r'C:\Users\Felicia Pangestu\Documents\BANGKIT\Capstone\splitedDataset\train_ds'

# Mendefinisikan kelas
CLASSES = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]

# Menghitung jumlah gambar per kelas dan total di setiap folder
test_class_counts, test_total_count = count_images_per_class_and_total(test_folder_path)
valid_class_counts, valid_total_count = count_images_per_class_and_total(valid_folder_path)
train_class_counts, train_total_count = count_images_per_class_and_total(train_folder_path)

# Menampilkan hasil
print("Jumlah gambar per kelas di folder test_ds:")
for class_name, count in test_class_counts.items():
    print(f"{class_name}: {count}")
print(f"Total gambar di folder test_ds: {test_total_count}\n")

print("Jumlah gambar per kelas di folder valid_ds:")
for class_name, count in valid_class_counts.items():
    print(f"{class_name}: {count}")
print(f"Total gambar di folder valid_ds: {valid_total_count}\n")

print("Jumlah gambar per kelas di folder train_ds:")
for class_name, count in train_class_counts.items():
    print(f"{class_name}: {count}")
print(f"Total gambar di folder train_ds: {train_total_count}\n")

# Menghitung total keseluruhan
total_overall = test_total_count + valid_total_count + train_total_count
print(f"Total keseluruhan dari ketiga folder: {total_overall}")

# Menampilkan jumlah total gambar per kelas
total_counts_per_class = defaultdict(int)
for class_name in CLASSES:
    total_counts_per_class[class_name] = (
        test_class_counts[class_name] +
        valid_class_counts[class_name] +
        train_class_counts[class_name]
    )

print("\nJumlah total gambar per kelas:")
for class_name, count in total_counts_per_class.items():
    print(f"{class_name}: {count}")
