import cv2
import numpy
import argparse
import sys
import imutils
from collections import deque
import time


# NOTE: If you pass cv.CHAIN_APPROX_NONE, all the boundary points are 
# stored. But actually do we need all the points? For eg, you found 
# the contour of a straight line. Do you need all the points on the 
# line to represent that line? No, we need just two end points of 
# that line. This is what cv.CHAIN_APPROX_SIMPLE does. It removes all 
# redundant points and compresses the contour, thereby saving memory.
# NOTE: https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
# NOTE: https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
# NOTE: https://www.youtube.com/watch?v=7FPD_UmFqqU


# TODO: make a scheduler to split the text, give each letter to 'generate_text',
# then draw it with the laser, pass the next letter and so on until the end
# of the string.
# IDEA: we can start from a given string and make her write it. I'll start
# from this.
def generate_text(to_draw):
    single_letter = numpy.zeros((800, 800), dtype = 'uint8')
    for letter in to_draw:
        # only 1 of line thickness, because this will not give problems of
        # 'double contour' of 'Prova_contorni'
        cv2.putText(single_letter, letter,
            (200, 600),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            10,
            (255, 255, 255),
            1)
        cv2.imshow('Letter', single_letter)
        
    return single_letter


#TENERE PREMUTO 'q' PER VEDERE I PUNTI DEL CONTORNO DELLA SCRITTA APPARIRE SULL'IMMAGINE
frame  = cv2.imread("Scritta1.jpg")
frame = cv2.resize(frame, (1000, 500))
frame = generate_text('M')
blur = cv2.GaussianBlur(frame, (7, 7), cv2.BORDER_DEFAULT)

canny = cv2.Canny(blur, 125, 175)
cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c = imutils.grab_contours(cnts)

if len(cnts) > 0:
    c = max(c, key=cv2.contourArea)
    for i in c:
        print(i)

i = 0
while True:
    cv2.imshow('Contours', canny)
    cv2.imshow('RGB_VideoFrame', frame)  

    key = cv2.waitKey(10)
    car = chr(key & 255)
    if car in ['a', 'A', chr(27)]:
        sys.exit(0)
    if car in ['q', 'Q', chr(27)]:   
        point = (c[i][0][0], c[i][0][1])
        print(len(point))
        print(point)
        
        cv2.circle(frame, point, 5, (255, 255, 0), -1)
        i += 1
