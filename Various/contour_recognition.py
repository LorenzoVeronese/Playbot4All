import cv2

# Load image, convert to grayscale, and perform Canny edge detection
image = cv2.imread('Template.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = 255 - cv2.Canny(gray, 120, 255, 1)

# Show image
cv2.imshow('canny', canny)
cv2.waitKey()
