import cv2 as cv

image_path = "images/t/IMG_20250408_100450_840.jpg"
image_path2 = "images/t/IMG_20250408_100450_897.jpg"
image_path3 = "images/t/IMG_20250408_100451_511.jpg"
image_path4 = "images/t/IMG_20250408_100451_621.jpg"

image = cv.imread(image_path)
print(image.shape)
cv.imshow("original image", image)

resized_image_10x10 = cv.resize(image, (10, 10))
# resized_image_50x50 = cv.resize(image, (50, 50))
# resized_image_100x100 = cv.resize(image, (100, 100))
# resized_image_1000x1000 = cv.resize(image, (1000, 1000))

cv.imshow("resized image", resized_image_10x10)

# image = cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
# cv.imshow("original image", image)
grey_image = cv.cvtColor(resized_image_10x10, cv.COLOR_BGR2GRAY)
_, binary_image_50 = cv.threshold(grey_image, 50, 255, cv.THRESH_BINARY)
# _, binary_image_100 = cv.threshold(grey_image, 100, 255, cv.THRESH_BINARY)
# _, binary_image_150 = cv.threshold(grey_image, 150, 255, cv.THRESH_BINARY)
# _, binary_image_200 = cv.threshold(grey_image, 200, 255, cv.THRESH_BINARY)
# _, binary_image_250 = cv.threshold(grey_image, 250, 255, cv.THRESH_BINARY)

cv.imshow("static threshold image", binary_image_50)
cv.waitKey(0)
