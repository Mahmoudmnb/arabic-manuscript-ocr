# import cv2 as cv
# import matplotlib.pyplot as plt
# import numpy as np

# # * read an image
# def read_image(path):
#     image = cv.imread(path)
#     cv.imshow("hi", image)
#     cv.waitKey(0)


# # * scale image
# def rescale(image, scale=0.7):
#     width = int(image.shape[0] * scale)
#     hight = int(image.shape[1] * scale)
#     dimension = (width, hight)
#     return cv.resize(image, dimension, interpolation=cv.INTER_AREA)


# image = cv.imread("mnb.png")
# resized_image = rescale(image, 0.2)
# cv.imshow("resized image", resized_image)
# cv.waitKey(1000)


# # * create a blank image
# blank_image = np.zeros((500, 500, 3), dtype="uint8")
# # * color some pixels
# # blank_image[200:300, 300:400] = 0, 255, 0
# # * draw a rectangle
# cv.rectangle(blank_image, (0, 0), (200, 200), (0, 255, 0), thickness=2)
# # * draw a circle
# cv.circle(
#     blank_image,
#     (blank_image.shape[0] // 2, blank_image.shape[1] // 2),
#     50,
#     color=(255, 0, 0),
#     thickness=3,
# )
# # * draw a line
# cv.line(blank_image, (100, 100), (300, 300), (0, 255, 255), 3, cv.LINE_AA, 0)
# # * put text in image
# cv.putText(
#     blank_image,
#     "hello world",
#     (0, blank_image.shape[1] // 2),
#     fontFace=cv.FONT_HERSHEY_PLAIN,
#     color=(40, 50, 60),
#     fontScale=5,
# )
# cv.imshow("blank image", blank_image)
# cv.waitKey(0)

# image = cv.imread("images/a.jpg")
# # * convert an image to grey scale
# grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# # * apply a blur to an image
# blurred_image = cv.GaussianBlur(image, (5, 5), cv.BORDER_DEFAULT)
# # * detect edges in the image
# canny_image = cv.Canny(image, 50, 50)
# # * crop an image
# cropped_image = image[0:200, 0:200]
# cv.imshow("image", cropped_image)
# cv.waitKey(0)


# # * translate image
# def translate(image, x, y):
#     transMat = np.float32([[1, 0, x], [0, 1, y]])
#     dimensions = (image.shape[1], image.shape[0])
#     return cv.warpAffine(image, transMat, dimensions)


# image = cv.imread("mnb.png")
# translated_image = translate(image, -50, 50)
# cv.imshow("translated image", translated_image)
# cv.waitKey(0)

# * rotate an image
# def rotate(image, angle, rotPoint=None):
#     (hight, width) = image.shape[:2]
#     if rotPoint is None:
#         rotPoint = (width // 2, hight / 2)
#     rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1)
#     return cv.warpAffine(image, rotMat, (width, hight))


# image = cv.imread("mnb.png")
# rotated_image = rotate(image, 90)
# cv.imshow("rotated_image", rotated_image)
# cv.waitKey(0)


# * flipping an image
# image = cv.imread("mnb.png")
# flipped_image = cv.flip(image, -1)
# cv.imshow("flipped_image", flipped_image)
# cv.waitKey(0)
# * contour
# image = cv.imread("images/a.jpg")
# grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# canny = cv.Canny(grey_image, 175, 175)
# contour, h = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# blank_image = np.zeros(image.shape[:2])
# contour_image = cv.drawContours(blank_image, contour, -1, (255, 255, 0))
# cv.imshow("grey image", contour_image)
# cv.waitKey(0)

# image = cv.imread("images/a.jpg")
# # * convert BGR image to HSV image
# HCV_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
# # * convert BGR image to LAB image
# LAB_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
# # * convert BGR image to RGB image
# RGB_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# # * convert BGR image to grey image
# grey_image = cv.cvtColor(HCV_image, cv.COLOR_BGR2GRAY)
# cv.imshow("image", RGB_image)
# cv.waitKey(0)

# * split an image to color channels
# image = cv.imread("images/a.jpg")
# b, g, r = cv.split(image)
# blank_image = np.zeros(r.shape, dtype="uint8")
# # cv.imshow("blue image", b)
# # cv.imshow("green image", g)
# # cv.imshow("red image", r)
# red_image = cv.merge([blank_image, blank_image, r])
# green_image = cv.merge([blank_image, g, blank_image])
# blue_image = cv.merge([b, blank_image, blank_image])
# # cv.imshow("blue image", blue_image)
# # cv.imshow("green image", green_image)
# # cv.imshow("red image", red_image)
# original_image = cv.merge([b, g, r])
# cv.imshow("original imag", original_image)
# cv.waitKey(0)
# * masked image
# image = cv.imread("images/a.jpg")
# blank_image = np.zeros(image.shape[:2], dtype="uint8")
# circle = cv.circle(
#     blank_image,
#     (blank_image.shape[0] // 2, blank_image.shape[1] // 2),
#     100,
#     color=(255, 0, 0),
#     thickness=-1,
# )
# masked_image = cv.bitwise_and(image, image, mask=circle)
# cv.imshow("masked imag", masked_image)
# cv.waitKey(0)
# ! important
# # * calculate histogram
# book_image = cv.imread("images/dataset/Book1/Book1_00000022_B.PNG")
# image = cv.imread("images/a.jpg")
# grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# histograms = cv.calcHist([grey_image], [0], None, [256], [0, 256])
# print(histograms[20])
# plt.figure()
# plt.title("histogram")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.plot(histograms)
# plt.xlim([0, 256])
# plt.show()
# # cv.imshow("grey_image", grey_image)
# # cv.imshow("image", image)
# cv.waitKey(0)

# # * Threeshoulding
# book_image = cv.imread("images/dataset/Book1/Book1_00000022_B.PNG")
# image = cv.imread("images/book.jpg", cv.IMREAD_GRAYSCALE)
# grey_image = cv.cvtColor(book_image, cv.COLOR_BGR2GRAY)
# # threshold, thr_img = cv.threshold(grey_image, 100, 255, cv.THRESH_BINARY)
# thr_img_adaptive = cv.adaptiveThreshold(
#     image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 5
# )
# cv.imshow("image", image)
# cv.imshow("grey_image", grey_image)
# # cv.imshow("binary_image", thr_img)
# cv.imshow("binary_image", thr_img_adaptive)
# cv.waitKey(0)
# # * read image as grey
# image = cv.imread("images/a.jpg", cv.IMREAD_GRAYSCALE)
# cv.imshow("binary_image", image)
# cv.waitKey(0)
