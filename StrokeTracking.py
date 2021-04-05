import cv2
import numpy
import sys

# IDEAS
# NOTE: canny and threshold+drawcontours are similar, but maybe
#   we can use the second one to record the most recent contours, so
#   that we know where is the last stroke (= where is the hand)

cam = cv2.VideoCapture(0)

while True:
    check, frame = cam.read()

    # CANNY METHOD
    # frame manipulation to get a more visible frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # NOTE: we have to try lots of values to see the best ones
    blur = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
    eroded = cv2.erode(blur, (3, 3), iterations = 3)
    canny = cv2.Canny(eroded, 125, 175)

    #cv2.imshow('Canny', canny)


    # CONTOURS METHOD
    # return all contours found
    contours, hierarchies = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))

    ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    blank = numpy.zeros(frame.shape, dtype = 'uint8')
    cv2.drawContours(blank, contours, -1, (255, 255, 255), 1)
    cv2.imshow('Contours', blank)


    # FIND LASER
    # this splits color channels: I take red, I find the position
    # of the pixel with max red value (red laser)
    b, g, r = cv2.split(frame)
    max_val = 0 # It's always 255
    max_col = 0 # NOTE: to see where to set these values by default
    max_row = 0
    for n_col, col in enumerate(r):
        for n_row, row in enumerate(col):
            if max_val < row:
                max_val = row
                max_col = n_col
                max_row = n_row
    print(max_col, max_row)
    cv2.imshow('Red channel', r)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
