import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.transform import resize

# Set parameters
IMAGE_SIZE = 256  # Resize all images and masks to 256x256
DATA_PATH = 'D:/OCTImages/OnNamaShivay/data/'  # Update this to your actual path
ANNOTATION_FILE = os.path.join(DATA_PATH, 'CNV ANNOT.json')  # JSON annotation file


def load_annotations(annotation_file):
    """
    Load annotations from the JSON file.
    """
    with open(annotation_file, 'r') as file:
        annotations = json.load(file)
    return annotations


def preprocess_data(image_folder, annotation_file):
    """
    Load all images and their masks from the specified data folder.
    """
    X = []  # Image data
    Y = []  # Mask data (annotations)

    # Load annotations
    annotations = load_annotations(annotation_file)

    # Iterate through subfolders (CNV, DME, NORMAL)
    for folder_name in os.listdir(image_folder):
        folder_path = os.path.join(image_folder, folder_name)
        if os.path.isdir(folder_path):  # Process only directories
            for image_filename in tqdm(os.listdir(folder_path), desc=f"Processing {folder_name}"):
                if image_filename.endswith('.jpeg'):
                    # Load and preprocess image
                    img_path = os.path.join(folder_path, image_filename)
                    img_file = cv2.imread(img_path)
                    img_file = resize(img_file, (IMAGE_SIZE, IMAGE_SIZE, 3))  # Resize image
                    img_arr = np.asarray(img_file) / 255.0  # Normalize image to [0, 1]
                    X.append(img_arr)

                    # Prepare corresponding mask from JSON annotations
                    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
                    annotation_key = image_filename.split('.')[0]  # Key to match in JSON file
                    if annotation_key in annotations:
                        regions = annotations[annotation_key]['regions']
                        for region in regions:
                            points = np.array(region['polygon'], dtype=np.int32)
                            cv2.fillPoly(mask, [points], (1))
                    Y.append(mask)

    # Convert to numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Split into train/test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test


# Call the preprocessing function
X_train, X_test, Y_train, Y_test = preprocess_data(DATA_PATH, ANNOTATION_FILE)

# Print dataset shapes
print(f"Training images: {X_train.shape}, Training masks: {Y_train.shape}")
print(f"Testing images: {X_test.shape}, Testing masks: {Y_test.shape}")
