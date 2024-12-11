import cv2
from matplotlib import pyplot as plt

def refractive_error():
    pass

def scaling():
    return cv2.resize(image, (128, 128), cv2.INTER_AREA)

def pre_processing():
    # Applying CLAHE
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    # Denoising the image
    denoised_image = cv2.bilateralFilter(enhanced_image, 9, 75, 75)

    # Normalization
    normalize_image = cv2.normalize(denoised_image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # Thresholding
    _, binary_img = cv2.threshold(normalize_image, 0.5, 1.0, cv2.THRESH_BINARY)

    return binary_img

# Load the image
image = cv2.imread("002.jpg")

refractive_error()
scaled_image = scaling()
pre_processed = pre_processing()

#Display the results
fig, axes = plt.subplots(1, 3, figsize=(15, 8))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image (RGB)")
axes[0].axis('off')

# Scaled image
axes[1].imshow(scaled_image, cmap='gray')
axes[1].set_title("Scaled Image")
axes[1].axis('off')

# Pre-processed image
axes[2].imshow(pre_processed, cmap='gray')
axes[2].set_title("Pre-processed Image")
axes[2].axis('off')

plt.tight_layout()
plt.show()
