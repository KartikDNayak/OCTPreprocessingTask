import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import pandas as pd

# Define the directories for training data and annotation files
TRAIN_DIR = "data"  # Replace with the path to your OCT images
ANNOTATION_FILES = ["data/CNV ANNOT.json",
    "data/labels_my-project-name_2024-11-14-11-53-40.json"]


def load_annotations(annotation_files):
    annotations = {}
    for file in annotation_files:
        with open(file, 'r') as f:
            data = json.load(f)
            annotations.update(data)  # Merge all annotation files
    return annotations


def preprocess_image(image_path):
    # Load and preprocess an image (e.g., resize, normalization)
    image = Image.open(image_path)
    image = image.resize((256, 256))  # Resize to 256x256 or any other size as required
    image = np.array(image)
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return image


def preprocess_mask(mask_path):
    # Load and preprocess a mask (e.g., resize, binary mask)
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale
    mask = mask.resize((256, 256))  # Resize to 256x256 or any other size as required
    mask = np.array(mask)
    mask = np.where(mask > 127, 1, 0)  # Threshold to binary mask
    return mask


def debug_annotations(annotations, dataset_dir):
    """Log mismatches between annotation keys and image filenames"""
    for key in annotations.keys():
        image_path = os.path.join(dataset_dir, key + ".png")  # Update extension if needed
        if not os.path.exists(image_path):
            print(f"Warning: No image found for {key}")


def preprocess_data(dataset_dir, annotation_files):
    # Load all annotations
    annotations = load_annotations(annotation_files)

    # Debugging: Check for mismatched annotation keys
    debug_annotations(annotations, dataset_dir)

    X = []  # Images
    Y = []  # Masks
    for key, annotation in annotations.items():
        image_path = os.path.join(dataset_dir, key + ".png")  # Change the extension if needed
        if not os.path.exists(image_path):
            print(f"No image found for {key}, skipping...")
            continue  # Skip if image does not exist

        mask_path = os.path.join(dataset_dir, key + "_mask.png")  # Assuming mask files are named similarly
        if not os.path.exists(mask_path):
            print(f"No mask found for {key}, skipping...")
            continue  # Skip if mask does not exist

        # Preprocess the image and mask
        image = preprocess_image(image_path)
        mask = preprocess_mask(mask_path)

        X.append(image)
        Y.append(mask)

    if not X or not Y:
        print("Error: No images or masks generated. Check annotation keys and filenames.")
        return None, None

    # Convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def main():
    # Define the directory containing your training data and annotation files
    X, Y = preprocess_data(TRAIN_DIR, ANNOTATION_FILES)

    if X is not None and Y is not None:
        print(f"Processed {len(X)} images and masks.")
        # You can now save X, Y to disk or feed them to a model
        # For example, you could use np.save to save the arrays for later use
        np.save("processed_images.npy", X)
        np.save("processed_masks.npy", Y)
    else:
        print("No valid data to process. Please check your annotations and filenames.")


if __name__ == "__main__":
    main()
