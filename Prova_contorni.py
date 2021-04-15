import cv2
import numpy
import argparse
import sys
import imutils
from collections import deque
import time

#TENERE PREMUTO 'q' PER VEDERE I PUNTI DEL CONTORNO DELLA SCRITTA APPARIRE SULL'IMMAGINE


frame  = cv2.imread("Foto_prova_contorni.jpg")
frame = cv2.resize(frame, (1000, 500))
hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv_img)

channel_h = h
channel_s = s
channel_v = v

minimum =51
maximum =219

(t, tmp) = cv2.threshold(
                channel_h,  # src
                maximum,  # threshold value
                0,  # we dont care because of the selected type
                cv2.THRESH_TOZERO_INV  # t type
            )

(t, channel_h) = cv2.threshold(
    tmp,  # src
    minimum,  # threshold value
    255,  # maxvalue
    cv2.THRESH_BINARY  # type
)

minimum = 32
maximum = 221

(t, tmp) = cv2.threshold(
                channel_s,  # src
                maximum,  # threshold value
                0,  # we dont care because of the selected type
                cv2.THRESH_TOZERO_INV  # t type
            )

(t, channel_s) = cv2.threshold(
    tmp,  # src
    minimum,  # threshold value
    255,  # maxvalue
    cv2.THRESH_BINARY  # type
)

minimum = 126
maximum = 255

(t, tmp) = cv2.threshold(
                channel_v,  # src
                maximum,  # threshold value
                0,  # we dont care because of the selected type
                cv2.THRESH_TOZERO_INV  # t type
            )

(t, channel_v) = cv2.threshold(
    tmp,  # src
    minimum,  # threshold value
    255,  # maxvalue
    cv2.THRESH_BINARY  # type
)

mask = cv2.bitwise_and(
    channel_h,
    channel_v
)
mask = cv2.bitwise_and(
    channel_s,
    mask
)

pts = deque(maxlen=64)
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

if len(cnts) > 0:
    c = max(cnts, key=cv2.contourArea)
    for i in c:

        print("\n")
        print(i)

cv2.namedWindow('LaserPointer')
cv2.resizeWindow('LaserPointer', 1000, 500)
cv2.moveWindow('LaserPointer', 0, 0)

cv2.namedWindow('RGB_VideoFrame')
cv2.resizeWindow('RGB_VideoFrame', 1000, 500)
cv2.moveWindow('RGB_VideoFrame', 0, 0)

i = 0
while True:
    cv2.imshow('RGB_VideoFrame', frame)
    cv2.imshow('LaserPointer', mask)

    key = cv2.waitKey(10)
    car = chr(key & 255)
    if car in ['a', 'A', chr(27)]:
        sys.exit(0)
    if car in ['q', 'Q', chr(27)]:
        point = (c[i][0][0], c[i][0][1])
        cv2.circle(frame, point, 5,
                   (0, 255, 255), 2)
        i = i + 1


