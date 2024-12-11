import cv2
from matplotlib import pyplot as plt
import numpy as np

def detect_cup(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    # Focus on the brightest regions (optic disc and cup)
    _, bright_regions = cv2.threshold(enhanced_image, 200, 255, cv2.THRESH_BINARY)

    # Morphological operations to isolate the optic disc and cup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bright_regions_refined = cv2.morphologyEx(bright_regions, cv2.MORPH_CLOSE, kernel)

    # Mask the original enhanced image to isolate the cup within the bright regions
    masked_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=bright_regions_refined)

    # Threshold within the optic disc to segment the cup (lighter center part)
    _, cup_segment = cv2.threshold(masked_image, 220, 255, cv2.THRESH_BINARY)

    # Further refine the cup region
    cup_refined = cv2.morphologyEx(cup_segment, cv2.MORPH_OPEN, kernel)

    return cup_refined

# Load the fundus image
image = cv2.imread("001.jpg")

if image is None:
    raise ValueError("Image could not be loaded. Check the file path.")

# Detect the cup region
cup_image = detect_cup(image)

# Display the results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Fundus Image")
axes[0].axis('off')

axes[1].imshow(cup_image, cmap='gray')
axes[1].set_title("Detected Cup Region")
axes[1].axis('off')

plt.tight_layout()
plt.show()
