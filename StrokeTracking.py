import cv2
import numpy
import sys

cam = cv2.VideoCapture(0)

while True:
    check, frame = cam.read()

    # frame manipulation to get a more visible frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred means sfocata. NOTE: We have to try lots of values, 
    # to see if this is useful
    blur = cv2.GaussianBlur(gray, (7, 7), cv2.BORDER_DEFAULT)
    eroded = cv2.erode(blur, (3, 3), iterations = 1)
    canny = cv2.Canny(blur, 125, 175)

    cv2.imshow('Video', canny)


    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
