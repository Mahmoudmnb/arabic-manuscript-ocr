import cv2
from scipy import ndimage  # Use SciPy for LoG
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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

# Generate a 1D Hanning window
# window_size = 900
# hanning_window_1d = np.hanning(window_size)
# cv.imshow("Hanning window 1 ", hanning_window_1d)

# # Generate a 2D Hanning window (can be used to create a filter kernel)
# hanning_window_2d = np.outer(hanning_window_1d, hanning_window_1d)
# cv.imshow("Hanning window 2 ", hanning_window_2d)
# # Normalize the 2D window to use as a simple averaging/blur filter kernel
# hanning_kernel = hanning_window_2d / np.sum(hanning_window_2d)

# gray = cv.imread(image_path2, cv.IMREAD_GRAYSCALE)
# print(gray.shape)
# cv.imshow("Gray", gray)


# # You could then convolve this kernel with an image using cv.filter2D
# # e.g., blurred_image = cv.filter2D(gray_image, -1, hanning_kernel)
# # blurredImage = cv.filter2D(gray, -1, hanning_kernel)
# # cv.imshow("Blurred with kernal", blurredImage)
# # However, Gaussian blur is far more common for general blurring tasks.

# # Plot the windows
# plt.figure(figsize=(10, 5))
# plt.subplot(121), plt.plot(hanning_window_1d), plt.title('1D Hanning Window')
# plt.subplot(122), plt.imshow(hanning_window_2d,
#                              cmap='viridis'), plt.title('2D Hanning Window')
# plt.show()

# print("Hanning window is mainly for signal processing (FFT) or filter design.")
# print("Direct spatial blurring usually uses Gaussian or Mean filters.")


def hanning_blur_opencv(image, kernel_size):
    """Applies a Hanning blur to an image using OpenCV.

    Args:
        image (numpy.ndarray): The input image (grayscale or color).
        kernel_size (int): The size of the Hanning kernel (must be odd).

    Returns:
        numpy.ndarray: The Hanning blurred image.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Generate 1D Hanning window
    hanning_1d = np.hanning(kernel_size)

    # Create 2D Hanning kernel (outer product)
    hanning_2d = np.outer(hanning_1d, hanning_1d)

    # Normalize the kernel so the sum of elements is 1
    hanning_2d /= np.sum(hanning_2d)

    # Apply the filter
    blurred_image = cv2.filter2D(
        image, -1, hanning_2d, borderType=cv2.BORDER_DEFAULT)

    return blurred_image



img = cv2.imread(image_path1)

if img is None:
    print("Error: Could not open or find the image.")
else:
    # Define the kernel size (odd number)
    blur_size = 7


    # Apply the Hanning blur
    hanning_blurred_image = hanning_blur_opencv(img, blur_size)

    # Display the original and blurred images
    cv2.imshow('Original Image', img)
    cv2.imshow('Hanning Blurred Image', hanning_blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


cv.waitKey(0)
