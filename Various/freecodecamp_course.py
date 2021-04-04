# https://www.youtube.com/watch?v=oXlwWbU8l2o&list=WL&index=166&t=3175s
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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



'''
# CONTOUR DETECTION
img = cv.imread('Template.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Second and third arguments of cv2.canny are our minVal and maxVal respectively.
canny = cv.Canny(img, 125, 175)

# contours: it's a list of coordinates of the contours found
# hierarchies: ex if you have a rectangle and inside it a square ecc
# CHAIN_...: says what type of approximation we want
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) #RETR_... according to what you want: external contours, all...
print(f'{len(contours)} contour(s) found')

# ANOTHER WAY of finding contours: THRESHOLD and then draw on BLANK background
# this binarizes the image
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)

blank = np.zeros(img.shape, dtype='uint8')
# now we want to draw contours on this blank image
cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow('Test', blank)
cv.waitKey(0)

# TIP: start using canny, then use threshold if you are not happy
'''



'''
# COLOR SPACES
# NOTE: when OpenCV displays an image, it assumes that it's a BGR, so if you
# pass it another format, you will see (using imshow) strange colours. For example
# if you convert an image BGR -> RGB OpenCV will show it as strange, but matplotlib, which
# is set on RGB, will show it correctly
img = cv.imread('Template.jpg')

# BGR -> grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# BGR -> HSV (based on how human perceive color)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#BGR -> LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

# other libraries doesn't use BGR format. ex matplotlib displays as RGB 
plt.imshow(img)

# BGR -> RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# you can make all inverse conversions, except for grayscale -> HSV (to make this
# you should do: grayscale -> BGR -> HSV)

# HSV -> BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('Test', hsv_bgr)
cv.waitKey(0)
'''



'''
# COLOR CHANNELS
img = cv.imread('Template.jpg')
# there are 3 color channels: blue, green, red
b, g, r = cv.split(img) # this splits the img into it's 3 colors
# this print in grayscale, which shows the distribution of color's
# intensities on every pixel (lighter = more concentrated)
cv.imshow('Test', b)
print(img.shape)
print(b.shape) # in effects, this has one shape: grayscale with imshow
print(g.shape)
print(r.shape)
cv.waitKey(0)

# let's merge the color channels together: you get the initial picture
merged = cv.merge([b, g, r])
cv.imshow('Test', merged)
cv.waitKey(0)

# and if I don't want to see a grayscale when I show the channel? Let's
# use a blank picture
blank = np.zeros(img.shape[:2], dtype = 'uint8')
blue = cv.merge([b, blank, blank]) # NOTE: the position of the blanks
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])
# as above: light = high distribution
cv.imshow('Test', red)
cv.waitKey(0)
'''



'''
# BLURRING TECHNIQUES
img = cv.imread('Template.jpg')
# we use to blur an image when it has lots of noise
# kernel/window = it's a window you draw over an image with size named "kernel size"
    # (ex if it's divided into 9 squares, it has kernel_size = 3x3). Blur is applied to
    # the middle pixer according to pixels which surround it
# AVERAGING
# we define a window and compute the single pixel as result of the ones which surrounds it
average = cv.blur(img, (3, 3)) # (3, 3) is the kernel size

# GAUSSIAN BLURING
# instead of compute the average, it gives a weight to all pixels and then computes the 
# product of all values. this appears as more natural
gauss = cv.GaussianBlur(img, (3, 3), 0)

# MEDIAN BLUR
# instead of finding the average, it finds the median (mediana)
# generally this method isn't good with large kernel sizes
median = cv.medianBlur(img, 3) # opencv assumes that this will be 3x3 simply giving it 3

# BILATERAL BLURING
# this consider contours. far pixels can influence
bilateral = cv.bilateralFilter(img, 5, 15, 15)
cv.imshow('Test', bilateral)
cv.waitKey(0)
'''



# BITWISE OPERATIONS
blank = np.zeros((400, 400), dtype = 'uint8')

rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

# AND (intersecting regions)
bitwise_and = cv.bitwise_and(rectangle, circle)

# OR (both intersecting and not-intersecting regions)
bitwise_or = cv.bitwise_or(rectangle, circle)

# XOR (for non-intersecting regions)
bitwise_xor = cv.bitwise_xor(rectangle, circle)

# NOT (invert)
bitwise_not = cv.bitwise_not(rectangle)
cv.imshow('Test', bitwise_not)
cv.waitKey(0)
