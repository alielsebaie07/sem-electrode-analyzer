import cv2
import numpy as np


def preprocess(image_path):
    # Load the image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check the image loaded correctly
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Otsu's thresholding to create binary image
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img, blurred, binary
