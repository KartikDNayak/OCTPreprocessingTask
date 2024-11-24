import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.transform import resize

IMAGE_SIZE = 256
MASK_SAVE_DIR = "data/masks"
PROCESSED_IMAGE_DIR = "data/processed_images"
TRAIN_DIR = "data"
ANNOTATION_FILES = [
    "data/CNV ANNOT.json",
    "data/labels_my-project-name_2024-11-14-11-53-40.json"
]

os.makedirs(MASK_SAVE_DIR, exist_ok=True)
os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)

def load_annotations(annotation_files):
    combined_annotations = {}
    for file in annotation_files:
        with open(file, 'r') as f:
            annotations = json.load(f)
            combined_annotations.update(annotations)
    print(f"Loaded {len(combined_annotations)} annotations")  # Debug
    return combined_annotations

def create_binary_mask(image_shape, annotations):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for annotation in annotations:
        if annotation.get("type") == "polygon":
            points = np.array(annotation["points"], dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
    print(f"Generated mask with non-zero pixels: {np.count_nonzero(mask)}")  # Debug
    return mask

def preprocess_data(image_folder, annotation_files):
    images = []
    masks = []
    annotations = load_annotations(annotation_files)

    for folder_name in os.listdir(image_folder):
        folder_path = os.path.join(image_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for file_name in tqdm(os.listdir(folder_path), desc=f"Processing {folder_name}"):
            file_path = os.path.join(folder_path, file_name)

            # Read and preprocess the image
            image = cv2.imread(file_path)
            if image is None:
                continue
            image = resize(image, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True)
            images.append(image)

            # Generate the mask
            base_name = os.path.splitext(file_name)[0]  # Strip extension
            if base_name in annotations:
                annotation_data = annotations[base_name]
                mask = create_binary_mask(image.shape, annotation_data)
                mask = resize(mask, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True)

                # Save mask
                mask_save_path = os.path.join(MASK_SAVE_DIR, f"{base_name}_mask.png")
                cv2.imwrite(mask_save_path, (mask * 255).astype(np.uint8))
                masks.append(mask)
            else:
                print(f"No annotation found for {base_name}")  # Debug

            # Save processed image
            processed_image_path = os.path.join(PROCESSED_IMAGE_DIR, file_name)
            cv2.imwrite(processed_image_path, (image * 255).astype(np.uint8))

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)

if len(X) == 0 or len(Y) == 0:
    print("Error: No images or masks generated. Check annotation keys and filenames.")
else:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"Training data: {X_train.shape}, {Y_train.shape}")
    print(f"Testing data: {X_test.shape}, {Y_test.shape}")
