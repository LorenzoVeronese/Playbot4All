import cv2
import numpy
import sys
import statistics
import imutils

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
    # LASER DETECTION
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


def hand_detection(frame):
    # HAND DETECTION
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    low_skin = numpy.array([0, 80, 100], dtype="uint8")  # 0, 48, 80
    up_skin = numpy.array([20, 200, 200], dtype="uint8")  # 20, 255, 255
    skin_mask = cv2.inRange(hsv, low_skin, up_skin)

    low_paper = numpy.array([100, 100, 100], dtype="uint8")  # 0, 48, 80
    up_paper = numpy.array([225, 225, 225], dtype="uint8")  # 20, 255, 255
    paper_mask = cv2.inRange(frame, low_paper, up_paper)

    cv2.imshow('Paper', paper_mask)

    # find moment and then centroid of the skin-based areas
    # NOTE: catch error when you have white light
    M = cv2.moments(skin_mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(frame, "centroid", (cX - 25, cY - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    blank = numpy.zeros((len(frame), len(frame[0])), dtype='uint8')
    circle = cv2.circle(
        blank,
        center_laser,
        100, (255, 255, 255), -1
    )
    bitwise_and = cv2.bitwise_and(skin_mask, circle)

    cv2.imshow('Skin Mask', skin_mask)
    #cv2.imshow('s', circle)
    cv2.imshow('Skin AND Laser', bitwise_and)
    cv2.imshow('Frame', frame)


cam = cv2.VideoCapture(0)

while True:
    check, frame = cam.read()
    
    # CANNY METHOD
    # frame manipulation to get a more visible frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # NOTE: we have to try lots of values to see the best ones
    blur = cv2.GaussianBlur(gray, (7, 7), cv2.BORDER_DEFAULT)
    eroded = cv2.erode(blur, (7, 7), iterations = 3)
    canny = cv2.Canny(blur, 125, 175)  # 125, 175

    cv2.imshow('Canny', canny)

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
    '''
    # LASER TRACKING
    center_laser = laser_tracking(frame)
    # the circle's center is at the mean value
    cv2.circle(
        frame, 
        center_laser, 
        100, (0, 255, 0), 4
    )
    #cv2.imshow('Normal', frame)
    '''

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
    '''
    # HAND DETECTION
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = numpy.array([0, 80, 100], dtype="uint8")#0, 48, 80
    upper = numpy.array([20, 200, 200], dtype="uint8")#20, 255, 255
    skin_mask = cv2.inRange(hsv, lower, upper)
    '''
    '''
    low_paper = numpy.array([100, 100, 100], dtype="uint8")  # 0, 48, 80
    up_paper = numpy.array([225, 225, 225], dtype="uint8")  # 20, 255, 255
    paper_mask = cv2.inRange(frame, low_paper, up_paper)

    cv2.imshow('Paper', paper_mask)
    '''

    '''
    # NOTE: catch error when you have white light
    M = cv2.moments(skin_mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(frame, "centroid", (cX - 25, cY - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    '''

    '''
    blank = numpy.zeros((len(frame), len(frame[0])), dtype='uint8')
    circle = cv2.circle(
        blank,
        center_laser,
        100, (255, 255, 255), -1
    )
    bitwise_and = cv2.bitwise_and(skin_mask, circle)
    '''
    # cv2.imshow('Skin Mask', skin_mask)
    # cv2.imshow('s', circle)
    # cv2.imshow('Skin AND Laser', bitwise_and)
    # cv2.imshow('Frame', frame)






    # PAPER DETECTION
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = numpy.array([105, 105, 105], dtype="uint8")  # 0, 48, 80
    upper = numpy.array([200, 200, 225], dtype="uint8")  # 20, 255, 255
    hsv_mask = cv2.inRange(frame, lower, upper)

    blank = numpy.zeros((len(frame), len(frame[0])), dtype='uint8')
    lower = numpy.array([95, 95, 100], dtype="uint8")  # 0, 48, 80
    upper = numpy.array([210, 210, 210], dtype="uint8")  # 20, 255, 255
    paper_mask = cv2.inRange(frame, lower, upper)

    bitwise_and = cv2.bitwise_and(hsv_mask, paper_mask)
    cnts, hierarchies = cv2.findContours(bitwise_and, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    cv2.drawContours(blank, cnts, -1, (255, 255, 255), 1)
    #cv2.imshow('HSV', hsv_mask)
    #cv2.imshow('Paper', paper_mask)
    cv2.imshow('Paper Contours', blank)


    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = numpy.int0(box)
        #Fist point of the box is le lowest, than we determine the seocond lowest and one on the top
        lower_point_1 = box[3]
        if box[0][1] > box[2][1]:
            lower_point_2 = box[0]
        else:
            lower_point_2 = box[2]
        ciao = cv2.line(frame, (lower_point_1[0], lower_point_1[1]),
                 (lower_point_2[0], lower_point_2[1]), (0, 255, 0), 2)
        cv2.imshow('box', ciao)


    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
