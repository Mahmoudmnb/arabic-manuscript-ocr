"""
SIFT feature-detection experiment.

This standalone script detects SIFT keypoints on sample images and displays
the keypoints overlaid on the selected image.
"""

import cv2 as cv

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
FACE_IMAGE_1 = "images/image_test1.jpg"
FACE_IMAGE_2 = "images/image_test2.jpg"
FACE_IMAGE_3 = "images/image_test3.jpg"


img = cv.imread(FACE_IMAGE_3)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

canny = cv.Canny(img, 125, 175)
cv.imshow("Canny image", canny)

sift = cv.SIFT_create()
kp = sift.detect(gray, None)

cv.drawKeypoints(img, kp, img)

cv.imshow("SIFT image", img)

cv.waitKey(0)
