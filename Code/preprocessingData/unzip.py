import os
import zipfile

target_directory = r"C:\Users\Felicia Pangestu\Documents\BANGKIT\Capstone"
os.makedirs(target_directory, exist_ok=True)
zip_file_path = os.path.join(target_directory, "../dataset-v2.zip")
print(f"Full path to zip file: {zip_file_path}")
extracted_directory = os.path.join(target_directory, "../dataset")
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_directory)
print(f"Dataset extracted to {extracted_directory}")