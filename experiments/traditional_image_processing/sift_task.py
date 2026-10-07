import numpy as np
import cv2 as cv

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
face_image1 = "images/image_test1.jpg"
face_image2 = "images/image_test2.jpg"
face_image3 = "images/image_test3.jpg"


img = cv.imread(face_image3)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

canny = cv.Canny(img, 125, 175)
cv.imshow("Canny image", canny)

sift = cv.SIFT_create()
kp = sift.detect(gray, None)

cv.drawKeypoints(img, kp, img)

cv.imshow('SIFT image', img)

cv.waitKey(0)
