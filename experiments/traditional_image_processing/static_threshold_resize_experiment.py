"""
Static thresholding experiment on resized sample images.

This script downsizes a sample image, converts it to grayscale, and applies
a fixed binary threshold for early preprocessing exploration.
"""

import cv2 as cv

IMAGE_PATH = "images/t/IMG_20250408_100450_840.jpg"
IMAGE_PATH_ALT_1 = "images/t/IMG_20250408_100450_897.jpg"
IMAGE_PATH_ALT_2 = "images/t/IMG_20250408_100451_511.jpg"
IMAGE_PATH_ALT_3 = "images/t/IMG_20250408_100451_621.jpg"
RESIZED_WIDTH = 10
RESIZED_HEIGHT = 10
STATIC_THRESHOLD = 50

image = cv.imread(IMAGE_PATH)
print(image.shape)
cv.imshow("original image", image)

resized_image_10x10 = cv.resize(image, (RESIZED_WIDTH, RESIZED_HEIGHT))
# resized_image_50x50 = cv.resize(image, (50, 50))
# resized_image_100x100 = cv.resize(image, (100, 100))
# resized_image_1000x1000 = cv.resize(image, (1000, 1000))

cv.imshow("resized image", resized_image_10x10)

# image = cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
# cv.imshow("original image", image)
grey_image = cv.cvtColor(resized_image_10x10, cv.COLOR_BGR2GRAY)
_, binary_image_50 = cv.threshold(grey_image, STATIC_THRESHOLD, 255, cv.THRESH_BINARY)
# _, binary_image_100 = cv.threshold(grey_image, 100, 255, cv.THRESH_BINARY)
# _, binary_image_150 = cv.threshold(grey_image, 150, 255, cv.THRESH_BINARY)
# _, binary_image_200 = cv.threshold(grey_image, 200, 255, cv.THRESH_BINARY)
# _, binary_image_250 = cv.threshold(grey_image, 250, 255, cv.THRESH_BINARY)

cv.imshow("static threshold image", binary_image_50)
cv.waitKey(0)
