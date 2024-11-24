# Import required libraries
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

# Set parameters
BASE_DIR = "D:/OCTimages/OnNamaShivay/data"
SUBFOLDERS = ["CNV", "DME", "NORMAL"]
JSON_FILES = [
    os.path.join(BASE_DIR, "CNV ANNOT.json"),
    os.path.join(BASE_DIR, "labels_my-project-name_2024-11-14-11-53-40.json"),
]
IMAGE_SIZE = 150

# Load JSON annotations
annotations = {}
for json_file in JSON_FILES:
    with open(json_file, "r") as file:
        data = json.load(file)
        annotations.update(data)


# Function to load and preprocess images
def load_images_and_labels(base_dir, subfolders):
    images = []
    labels = []
    label_map = {"NORMAL": 0, "CNV": 1, "DME": 2}

    for folder in subfolders:
        folder_path = os.path.join(base_dir, folder)
        for img_file in tqdm(os.listdir(folder_path), desc=f"Loading {folder}"):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                images.append(img)
                labels.append(label_map[folder])

    return np.array(images), np.array(labels)


# Load images and labels
images, labels = load_images_and_labels(BASE_DIR, SUBFOLDERS)

# Normalize images
images = images / 255.0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=3)
y_val = to_categorical(y_val, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Initialize data augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)

print("Class Weights:", class_weights)
print("Preprocessing completed.")
