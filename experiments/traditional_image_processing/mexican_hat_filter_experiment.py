"""
Mexican Hat / Laplacian-of-Gaussian filter experiment.

This standalone script compares OpenCV Gaussian blur and SciPy Gaussian
filtering before applying a Laplacian filter to a manuscript image.
"""

import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, laplace

SAMPLE_BOOK4_PAGE = "images/dataset/Book4/Book4_00000207_B.png"
SAMPLE_BOOK6_PAGE = "images/dataset/Book6/Book6_00000029_B.PNG"
SPOT_WITH_MISSING_LETTERS = "images/dataset/Book1/Book1_00000022_B.PNG"
SPOT_WITH_MISSING_LETTERS_ALT = "images/dataset/Book1/Book1_00000126_B.PNG"
MISSING_LETTERS_PAGE = "images/dataset/Book5/Book5_00000075_B.PNG"
MISSING_LETTERS_PAGE_ALT = "images/dataset/Book3/Book3_00000183_B.PNG"
MISSING_LETTERS_PAGE_ALT_2 = "images/dataset/Book1/Book1_00000182_A.PNG"
PAGE_WITH_SHADOW = "images/dataset/Book3/Book3_00000183_A.PNG"
PAGE_WITH_SHADOW_ALT = "images/dataset/Book2/Book2_000122_A.PNG"
BOLD_LETTERS_PAGE = "images/dataset/Book7/Book7_00000336_B.png"

# The Mexican Hat filter is closely related to the Laplacian of Gaussian.

# Load an example image
image_array = cv.imread(SAMPLE_BOOK4_PAGE, cv.IMREAD_GRAYSCALE)


# OpenCV GaussianBlur comparison.
blurred_image = cv.GaussianBlur(image_array.copy(), (11, 11), 0)

cv.imshow("GaussianBlur", blurred_image)

# Apply Gaussian blur
blurred_image = gaussian_filter(image_array.copy(), sigma=2)

cv.imshow("gaussian_filter", blurred_image)
# blurred_image = image_array.copy()


# Apply Laplacian operator
mexican_hat_filtered = laplace(blurred_image)

cv.imshow("mexican_hat_filtered", mexican_hat_filtered)

# Display the original and filtered images
plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(image_array, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(mexican_hat_filtered, cmap='gray')
# plt.title('Mexican Hat Filtered Image (Manual)')
# plt.axis('off')

# plt.show()


cv.waitKey(0)
