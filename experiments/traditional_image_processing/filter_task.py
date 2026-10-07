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

# Load image, convert to grayscale, and find edges (Hough often works on edges)
originalImage = cv.imread(image_path1)


if originalImage is None:
    print("Error: Image not loaded. Check the path.")
    exit()


# ? HoughLines

gray = cv.cvtColor(originalImage, cv.COLOR_BGR2GRAY)
# Use Canny to get edges first
edges = cv.Canny(gray, 50, 150, apertureSize=3)

# Perform Hough Line Transform (Probabilistic version is often more efficient)
# arguments: edge map, rho resolution, theta resolution, threshold (min votes)
lines = cv.HoughLinesP(edges, 1, np.pi / 180,
                       threshold=100, minLineLength=50, maxLineGap=10)


# Draw the detected lines on the original image
line_image = originalImage.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_image, (x1, y1), (x2, y2),
                (0, 0, 255), 2)  # Draw red lines

# Display the result (using matplotlib for inline display if needed)
# plt.figure(figsize=(10, 5))
# plt.subplot(121), plt.imshow(cv.cvtColor(
#     original_image, cv.COLOR_BGR2RGB)), plt.title('Original Image')
# plt.subplot(122), plt.imshow(cv.cvtColor(
#     line_image, cv.COLOR_BGR2RGB)), plt.title('Hough Lines Detected')
# plt.show()

# Or use cv.imshow if not in a notebook environment
cv.imshow('Original', originalImage)
cv.imshow('Edges', edges)
cv.imshow('Hough Lines', line_image)


# # ______________________________________________________________

# Generate a 1D Hanning window
window_size = 400
hanning_window_1d = np.hanning(window_size)
# cv.imshow("Hanning window 1 ", hanning_window_1d)

# Generate a 2D Hanning window (can be used to create a filter kernel)
hanning_window_2d = np.outer(hanning_window_1d, hanning_window_1d)
cv.imshow("Hanning window 2 ", hanning_window_2d)
# Normalize the 2D window to use as a simple averaging/blur filter kernel
hanning_kernel = hanning_window_2d / np.sum(hanning_window_2d)

# You could then convolve this kernel with an image using cv.filter2D
# e.g., blurred_image = cv.filter2D(gray_image, -1, hanning_kernel)
# blurredImage = cv.filter2D(gray, -1, hanning_kernel)
# cv.imshow("Blurred with kernal", blurredImage)
# However, Gaussian blur is far more common for general blurring tasks.

# Plot the windows
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.plot(hanning_window_1d), plt.title('1D Hanning Window')
plt.subplot(122), plt.imshow(hanning_window_2d,
                             cmap='viridis'), plt.title('2D Hanning Window')
# plt.show()

print("Hanning window is mainly for signal processing (FFT) or filter design.")
print("Direct spatial blurring usually uses Gaussian or Mean filters.")
# # ________________________________________________________________________

# # ________________________________________________________________________


# Load image and convert to grayscale
# Use an image with small light/dark details
image = cv.imread(image_path1)
if image is None:
    print("Error: Image not loaded. Check the path.")
    exit()

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Define a structuring element (e.g., a 5x5 rectangle or ellipse)
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))


# Apply Top Hat
tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)

# Apply Black Hat
blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)

#! this line for just convert every white to black and every black to white
_, blackhat = cv.threshold(blackhat, 150, 255, cv.THRESH_BINARY_INV)
# Display results
# plt.figure(figsize=(15, 5))
# plt.subplot(131), plt.imshow(gray, cmap='gray'), plt.title('Grayscale Image')
# plt.subplot(132), plt.imshow(tophat, cmap='gray'), plt.title(
#     'Top Hat (Bright Details)')
# plt.subplot(133), plt.imshow(blackhat, cmap='gray'), plt.title(
#     'Black Hat (Dark Details)')
# # plt.show()

# Or use cv.imshow
cv.imshow('Grayscale', gray)
cv.imshow('Top Hat', tophat)
cv.imshow('Black Hat', blackhat)
# cv.waitKey(0)
# cv.destroyAllWindows()
# # _________________________________________________________________________


# # _________________________________________________________________________


# Load image and convert to grayscale
image = cv.imread(image_path1)  # Use a suitable image
if image is None:
    print("Error: Image not loaded. Check the path.")
    exit()

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(
    float)  # Convert to float for filtering

# Apply Laplacian of Gaussian (Mexican Hat)
# sigma controls the size of the 'hat' (scale of features)
sigma = 5
log_filtered = ndimage.gaussian_laplace(gray, sigma=sigma)

# The result highlights edges and blobs. Zero-crossings indicate edges.
# Often you might look for zero crossings or local extrema in the result.

# Display the result (LoG output can be positive or negative)
# plt.figure(figsize=(10, 5))
# plt.subplot(121), plt.imshow(gray, cmap='gray'), plt.title('Grayscale Image')
# # Display LoG result - may need normalization or specific colormap for clarity
# plt.subplot(122), plt.imshow(log_filtered, cmap='coolwarm'), plt.title(
#     f'Laplacian of Gaussian (sigma={sigma})')
# plt.colorbar()  # Show the range of values
# plt.show()

# # Alternative using OpenCV: GaussianBlur then Laplacian
# Kernel size derived from sigma
blurred = cv.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
# Use 64F for accuracy (allows negative values)
log_cv = cv.Laplacian(blurred, cv.CV_64FC1)
cv.imshow("Laplacian", log_cv)
# plt.imshow(log_cv, cmap='coolwarm'), plt.title(f'LoG via OpenCV (sigma={sigma})')
# plt.show()

# # _________________________________________________________________________


cv.waitKey(0)
cv.destroyAllWindows()
