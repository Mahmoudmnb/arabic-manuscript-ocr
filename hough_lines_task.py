import cv2 as cv
import numpy as np


# ? Hough Lines detection of image and there is two way to detect the lines
# ? first is HoughLines :
# ? it's gives the lines base on rho (the distance between top-left point to the line) and theta
#
# ? and second is HoughLinesP : (more specific)
# ? gives all lines with (x1, y1) to (x2, y2)


origianlImg = cv.imread("images/a.jpg")
# resizeRate = float(input("input the resize rate of image: "))
resizeRate = 1.5
origianlImg = cv.resize(
    origianlImg, (int(origianlImg.shape[1] * resizeRate), int(origianlImg.shape[0] * resizeRate)))
houghLinesimg = origianlImg.copy()


# ! Hough Lines
# ? Starting initialization for HoughLines
gray = cv.cvtColor(houghLinesimg, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray.copy(), 50, 150, apertureSize=3)

cv.imshow("Canny", edges)
# cv.imshow("Binary", binary)

# it gives the lines base on
lines = cv.HoughLines(edges, 1, np.pi/180, 200)

if (lines is not None):
    for line in lines:
        rho, theta = line[0]
        print(theta)
        print(rho)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv.line(houghLinesimg, (x1, y1), (x2, y2), (0, 0, 255), 2)

lines = cv.HoughLinesP(edges, 1, np.pi/180, 50,
                       minLineLength=100, maxLineGap=10)
if (lines is not None):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(houghLinesimg, (x1, y1), (x2, y2), (0, 255, 0), 2)


cv.imshow("Original image", origianlImg)
cv.imshow('Hough Lines base on Canny', houghLinesimg)

cv.waitKey(0)
