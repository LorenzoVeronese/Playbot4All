import cv2
import numpy
import sys
import statistics

# IDEAS
# NOTE: canny and threshold+drawcontours are similar, but maybe
#   we can use the second one to record the most recent contours, so
#   that we know where is the last stroke (= where is the hand)
# NOTE: 
#   I know where the laser is
#   I consider a circle around that point
#   in that circle I want the pen(low brightness) and the hand(pink/many shadows)


# TODO:
#   know if the hand/pen are in the circle around the laser


def laser_tracking(frame):
    # FIND LASER
    # this splits color channels: I take red, I find the position
    # of the pixel with max red value (red laser)
    # NOTE: the red light in the hand distracts this tracker, which
    # start to go on the hand and not on the laser. So I decided
    # to use also brightness (see when I use h) and then take the mean
    # as the center value
    b, g, r = cv2.split(frame)

    # finds max's coordinates
    laser_index = numpy.unravel_index(numpy.argmax(r, axis=None), r.shape)
    '''
    # finds max's coordinates (more red), but too slow.
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
    '''

    # same as for red
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(frame)
    bright_index = numpy.unravel_index(numpy.argmax(h, axis=None), h.shape)

    rows = [laser_index[1], bright_index[1]]
    cols = [laser_index[0], bright_index[0]]

    return (int(statistics.mean(rows)), int(statistics.mean(cols)))


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


    '''
    # CONTOURS METHOD
    # return all contours found
    contours, hierarchies = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))

    ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    blank = numpy.zeros(frame.shape, dtype = 'uint8')
    cv2.drawContours(blank, contours, -1, (255, 255, 255), 1)
    cv2.imshow('Contours', blank)
    '''

    # LASER TRACKING
    center = laser_tracking(frame)
    # the circle's center is at the mean value
    cv2.circle(
        frame, 
        center, 
        100, (0, 255, 0), 4
    )
    cv2.imshow('Normal', frame)


    '''
    # NOTE: we can use matplotlib to see which values are the best 
    # FIND BLACK
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(frame)
    min_val = 255
    for n_col, col in enumerate(v):
        for n_row, row in enumerate(col):
            if min_val > row:
                min_val = row
                min_col = n_col
                min_row = n_row
    print(min_val)
    print(min_col, min_row)
    cv2.imshow('HSV', v)
    '''

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
