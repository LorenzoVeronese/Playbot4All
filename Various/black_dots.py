# https://www.geeksforgeeks.org/white-and-black-dot-detection-using-opencv-python/
# NOTE: this doesn't work!
import cv2
import numpy

pic = "Template.jpg"
# convert to grayscale
gray_pic = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
# threshold
th, threshed = cv2.threshold(
    gray_pic, 
    100, 255, 
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
)

# contours
contours = cv2.findContours(threshed, 
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE)[-2]

xcnts = []
# !!! select a certain min and max area of the stroke
for contour in contours:
    if 1 < cv2.contourArea(contour) < 50:
        xcnts.append(contour)
image = np.uint8(contours)
cv2.imshow('canny', image)

