import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
image_dir = "insect_pest_dataset/Images"
label_dir = "insect_pest_dataset/Annotations"
output_dir = "data"

# Create dirs
os.makedirs(f"{output_dir}/train/images", exist_ok=True)
os.makedirs(f"{output_dir}/train/labels", exist_ok=True)
os.makedirs(f"{output_dir}/val/images", exist_ok=True)
os.makedirs(f"{output_dir}/val/labels", exist_ok=True)

# List all files
images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
labels = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in images]

# Split (80% train, 20% val)
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Copy files
for img, lbl in zip(train_images, train_labels):
    shutil.copy(f"{image_dir}/{img}", f"{output_dir}/train/images/{img}")
    shutil.copy(f"{label_dir}/{lbl}", f"{output_dir}/train/labels/{lbl}")

for img, lbl in zip(val_images, val_labels):
    shutil.copy(f"{image_dir}/{img}", f"{output_dir}/val/images/{img}")
    shutil.copy(f"{label_dir}/{lbl}", f"{output_dir}/val/labels/{lbl}")

print("Dataset split complete!")