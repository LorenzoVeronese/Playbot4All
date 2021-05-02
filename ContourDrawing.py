import cv2
import numpy
import argparse
import sys
import imutils
from collections import deque
import time
import re


# NOTE: If you pass cv.CHAIN_APPROX_NONE, all the boundary points are 
# stored. But actually do we need all the points? For eg, you found 
# the contour of a straight line. Do you need all the points on the 
# line to represent that line? No, we need just two end points of 
# that line. This is what cv.CHAIN_APPROX_SIMPLE does. It removes all 
# redundant points and compresses the contour, thereby saving memory.
# NOTE: https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
# NOTE: https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
# NOTE: https://www.youtube.com/watch?v=7FPD_UmFqqU


# TODO: make a scheduler to split the text, give each letter to 'generate_letter',
# then draw it with the laser, pass the next letter and so on until the end
# of the string.
# IDEA: we can start from a given string and make her write it. I'll start
# from this.


class ContourDrawer(object):
    def __init__(self, file_name='Input.txt'):
        self.file_name = file_name
        # file_type: text or picture
        if file_name.split('.', 1)[1] == 'txt':
            self.file_type = 'txt'
        else:
            self.file_type = 'pic'

        # what to draw: read file if txt, read picture if pic
        if self.file_type == 'txt':
            self.to_draw = self.read_text_file()
            self.current_draw = None
        else:
            self.to_draw = cv2.imread(file_name)
            self.current_draw = self.to_draw


    def read_text_file(self):
        """
        read the image to draw is from a text file
        """
        fd = open(self.file_name, 'r')
        
        text = fd.readline()
        if text[len(text)-1] == '\n':
            text == text[0: len(text)-1]

        return text
        

    def generate_letter(self):
        """
        create the picture with the letter to draw at the center
        """
        # if I've finished to draw, return None type
        if self.to_draw == '':
            return None

        self.current_draw = numpy.zeros((800, 800), dtype='uint8')
        cv2.putText(self.current_draw, self.to_draw[0], (200, 600),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 10, (255, 255, 255), 1)

        # remove the first letter, so then I'll draw the next one
        # (see while loop in 'self.run')
        self.to_draw = self.to_draw[1:]
        print(self.to_draw)


    def draw(self):
        """
        find contours and give the contour's point list
        """
        frame  = self.current_draw.copy()
        frame = cv2.resize(frame, (1000, 500))
        # blur = cv2.GaussianBlur(frame, (7, 7), cv2.BORDER_DEFAULT)

        canny = cv2.Canny(frame, 125, 175)
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

    
    def run(self):
        """
        run from here
        """
        if self.file_type == 'txt':
            while self.to_draw != None:
                self.generate_letter()
                self.draw()
        elif self.file_type == 'pic':
            self.draw()
        else:
            exit(0)


if __name__ == '__main__':
    drawer = ContourDrawer()
    drawer.run()