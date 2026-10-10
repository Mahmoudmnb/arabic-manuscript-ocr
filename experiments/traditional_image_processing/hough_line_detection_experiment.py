"""
Hough-line detection experiment for manuscript and page images.

This standalone OpenCV script compares standard Hough lines and
probabilistic Hough lines on a sample image.
"""

import cv2 as cv
import numpy as np


IMAGE_PATH = "images/a.jpg"
RESIZE_RATE = 1.5


original_image = cv.imread(IMAGE_PATH)
# resizeRate = float(input("input the resize rate of image: "))
original_image = cv.resize(
    original_image,
    (
        int(original_image.shape[1] * RESIZE_RATE),
        int(original_image.shape[0] * RESIZE_RATE),
    ),
)
hough_lines_image = original_image.copy()


# Hough line detection works on the Canny edge map.
gray = cv.cvtColor(hough_lines_image, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray.copy(), 50, 150, apertureSize=3)

cv.imshow("Canny", edges)
# cv.imshow("Binary", binary)

# Standard Hough lines are represented by rho and theta.
lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        print(theta)
        print(rho)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv.line(hough_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Probabilistic Hough lines return explicit endpoint coordinates.
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=10)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(hough_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)


cv.imshow("Original image", original_image)
cv.imshow("Hough Lines base on Canny", hough_lines_image)

cv.waitKey(0)
