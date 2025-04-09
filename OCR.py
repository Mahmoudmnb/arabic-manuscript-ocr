import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

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

image_path = "images/book.jpg"

original_image = cv.imread(image_path)
original_image = cv.resize(
    original_image, (original_image.shape[1] // 3, original_image.shape[0] // 3)
)
grey_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
cv.imshow("original image", original_image)
cv.imshow("grey image", grey_image)

# ? Threeshoulding with static threshold value

# _, binary_image = cv.threshold(grey_image, 90, 255, cv.THRESH_BINARY)
# cv.imshow("static threshold image", binary_image)


# ? adaptive threeshoulding

thr_img_adaptive = cv.adaptiveThreshold(
    grey_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 5
)
cv.imshow("adaptive threshold image", thr_img_adaptive)


# ?OTSU threshold image

# blur = cv.GaussianBlur(grey_image, (3, 3), 0)
# ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# cv.imshow("OTSU image", th3)


# ? canny image threshold

# canny_image = cv.Canny(grey_image, 200, 200)
# blur = cv.GaussianBlur(canny_image, (3, 3), 0)
# ones = np.ones((3, 3))
# image = cv.morphologyEx(blur, cv.MORPH_ERODE, ones)
# _, binary_image = cv.threshold(image, 10, 255, cv.THRESH_BINARY_INV)
# cv.imshow("canny image", binary_image)

histograms = cv.calcHist([grey_image], [0], None, [256], [0, 256])
plt.figure()
plt.title("histogram")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(histograms)
plt.xlim([0, 256])
plt.show()

# max_value = np.max(histograms)
# beak_index = np.where(histograms == max_value)[0]
# while beak_index > 0:
#     if histograms[beak_index - 5] < histograms[beak_index]:
#         beak_index -= 1
#     else:
#         break
# print(beak_index)
# print(histograms[beak_index])
cv.waitKey(0)

# ! yolo model for object detection

# ! contore
