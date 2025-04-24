import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter, laplace
import matplotlib.pyplot as plt

# ? "Mexican Hat blur" is closely related to the Laplacian of Gaussian (LoG) filter.

image_path1 = "images/dataset/Book4/Book4_00000207_B.png"
image_path2 = "images/dataset/Book6/Book6_00000029_B.PNG"
spot_with_missing_letters = "images/dataset/Book1/Book1_00000022_B.PNG"
spot_with_missing_letters1 = "images/dataset/Book1/Book1_00000126_B.PNG"
missing_letters = "images/dataset/Book5/Book5_00000075_B.PNG"
missing_letters1 = "images/dataset/Book3/Book3_00000183_B.PNG"
missing_letters2 = "images/dataset/Book1/Book1_00000182_A.PNG"
with_shadow = "images/dataset/Book3/Book3_00000183_A.PNG"
with_shadow1 = "images/dataset/Book2/Book2_000122_A.PNG"
bold_letters = "images/dataset/Book7/Book7_00000336_B.png"

# Load an example image
image_array = cv.imread(image_path1, cv.IMREAD_GRAYSCALE)


# ! its equal for gaussian_filter
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
