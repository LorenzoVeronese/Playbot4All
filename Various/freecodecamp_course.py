import cv2 as cv
import numpy as np

'''
# PICTURE
img = cv.imread('Template.jpg')
cv.imshow('Template', img)
cv.waitKey(0) # close the pic's window if I press '0'
'''


'''
# RESIZE AND RESCALE
# for every type of video
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# for live videos
def changeResolution(width, height):
    capture.set(3, width)
    capture.set(4, height)
'''


'''
# VIDEO
capture = cv.VideoCapture(0) # 0 for cam, path for video
# read video frame by frame
while True:
    isTrue, frame = capture.read() # succesfully readed?, frame captured
    frame_resized = rescaleFrame(frame)
    # show frame by frame
    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cv.destroyAllWindows()
'''


'''
# DRAW ON PICTURE
# initialize a blank picture
blank = np.zeros((500, 500, 3), dtype = 'uint8') # (height, width, number of color channels), type

# paint the image a certain color
blank[:] = 0, 0, 255 # red
# paint a certain area
blank[200:300, 300:400] = 0, 255, 0

# draw a rectangle
cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness = 2) # thickness = cv.FILLED will fill the rectangle
# cv.rectangle(blank, (0, 0), (blank.shape[1] // 2, blank.shape[0] // 2), (0, 255, 0), thickness=2)

# draw a circle
# center, radius ...
cv.circle(blank, (250, 250), 40, (255, 0, 255), thickness = 3)

# draw a line
cv.line(blank, (0, 0), (250, 250), (255, 255, 255), thickness=2)

# draw text
cv.putText(blank, 'Hello', (255, 255), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
cv.imshow('Blank', blank)
cv.waitKey(0)
'''


'''
# BASIC FUNCTIONS
# converting to grayscale
img = cv.imread('Template.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# blur an image (eliminate some noise)
blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)

# find edges
canny = cv.Canny(blur, 125, 175) # I've passed blur!

# dilating an image
dilated = cv.dilate(canny, (3, 3), iterations = 1)

# eroding
eroded = cv.erode(dilated, (3, 3), iterations = 1)

# resize
resized = cv.resize(img, (500, 500), interpolation = cv.INTER_AREA) # INTER_CUBIC is the slowest

# crop
cropped = img[50:200, 200:400]
cv.imshow('Test', cropped)
cv.waitKey(0)
'''


'''
# TRANSFORMATIONS
# translation
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

img = cv.imread('Template.jpg')
translated = translate(img, 100, 100)

# rotation
def rotate(img, angle, rotPoint = None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width // 2, height // 2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

img = cv.imread('Template.jpg')
rotated = rotate(img, 45)
rotated_rotated = rotate(rotated, 45)

# resizing 
resized = cv.resize(img, (500, 500), interpolation = cv.INTER_CUBIC) #interpolation depends on what ypu do 

# flipping
flip = cv.flip(img, -1) #0 = flip vertically, 1 = orizzontally, -1 = both

# cropping
cropped = img[200:400, 300:400]
cv.imshow('Test', cropped)
cv.waitKey(0)
'''



# CONTOUR DETECTION
img = cv.imread('Template.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Second and third arguments of cv2.canny are our minVal and maxVal respectively.
canny = cv.Canny(img, 125, 175)

cv.imshow('Test', canny)
cv.waitKey(0)
