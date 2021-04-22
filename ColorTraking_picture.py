import cv2
#bla bal bal
import numpy
import argparse
import sys
import imutils
from collections import deque

OBJECT_LENGHT = (16 / 2.54) * 96 #(cm/(inches * cm)) * (pixel*inches) (resolution

def empty(a):
    pass
#Create the track bar HSV values (hue, saturation, value = brightness) to default
cv2.namedWindow("Prova")
cv2.resizeWindow("Prova", 640, 240)
cv2.createTrackbar("Hue MIN", "Prova", 17, 255, empty)
cv2.createTrackbar("Hue MAX", "Prova", 65, 255, empty)
cv2.createTrackbar("Sat MIN", "Prova", 95, 255, empty)
cv2.createTrackbar("Sat MAX", "Prova", 255, 255, empty)
cv2.createTrackbar("Val MIN", "Prova", 111, 255, empty)
cv2.createTrackbar("Val MAX", "Prova", 255, 255, empty)

class LaserTracker(object):
        def values(self):
            self.hue_min = cv2.getTrackbarPos("Hue MIN", "Prova")
            self.hue_max = cv2.getTrackbarPos("Hue MAX", "Prova")
            self.sat_min = cv2.getTrackbarPos("Sat MIN", "Prova")
            self.sat_max = cv2.getTrackbarPos("Sat MAX", "Prova")
            self.val_min = cv2.getTrackbarPos("Val MIN", "Prova")
            self.val_max = cv2.getTrackbarPos("Val MAX", "Prova")

        def __init__(self, cam_width=640, cam_height=480, hue_min=20, hue_max=120,
                 sat_min=100, sat_max=255, val_min=200, val_max=256,
                 display_thresholds=False):

            #Initial values
            self.cam_width = cam_width
            self.cam_height = cam_height
            self.hue_min = hue_min
            self.hue_max = hue_max
            self.sat_min = sat_min
            self.sat_max = sat_max
            self.val_min = val_min
            self.val_max = val_max
            self.display_thresholds = display_thresholds

            self.capture = None  # camera capture device
            self.channels = {
                'hue': None,
                'saturation': None,
                'value': None,
                'laser': None,
            }

            self.previous_position = None
            self.trail = numpy.zeros((self.cam_height, self.cam_width, 3),numpy.uint8)

        def create_and_position_window(self, name, xpos, ypos):
            """Creates a named widow placing it on the screen at (xpos, ypos)."""
            # Create a window
            cv2.namedWindow(name)
            # Resize it to the size of the camera image
            cv2.resizeWindow(name, self.cam_width, self.cam_height)
            # Move to (xpos,ypos) on the screen
            cv2.moveWindow(name, xpos,  ypos)

        def setup_camera_capture(self, device_num=0):
            """Perform camera setup for the device number (default device = 0).
            Returns a reference to the camera Capture object.
            """
            try:
                device = int(device_num)
                sys.stdout.write("Using Camera Device: {0}\n".format(device))
            except (IndexError, ValueError):
                # assume we want the 1st device
                device = 0
                sys.stderr.write("Invalid Device. Using default device 0\n")

            # Try to start capturing frames
            self.capture = cv2.VideoCapture(device)
            if not self.capture.isOpened():
                sys.stderr.write("Failed to Open Capture device. Quitting.\n")
                sys.exit(1)

            # set the wanted image size from the camera
            self.capture.set(
                cv2.cv.CV_CAP_PROP_FRAME_WIDTH if cv2.__version__.startswith('2') else cv2.CAP_PROP_FRAME_WIDTH,
                self.cam_width
            )
            self.capture.set(
                cv2.cv.CV_CAP_PROP_FRAME_HEIGHT if cv2.__version__.startswith('2') else cv2.CAP_PROP_FRAME_HEIGHT,
                self.cam_height
            )
            return self.capture

        def handle_quit(self, delay=10):
            """Quit the program if the user presses "Esc" or "q"."""
            key = cv2.waitKey(delay)
            c = chr(key & 255)
            if c in ['c', 'C']:
                self.trail = numpy.zeros((self.cam_height, self.cam_width, 3),
                                         numpy.uint8)
            if c in ['q', 'Q', chr(27)]:
                sys.exit(0)

        def threshold_image(self, channel):
            if channel == "hue":
                minimum = self.hue_min
                maximum = self.hue_max
            elif channel == "saturation":
                minimum = self.sat_min
                maximum = self.sat_max
            elif channel == "value":
                minimum = self.val_min
                maximum = self.val_max

            (t, tmp) = cv2.threshold(
                self.channels[channel],  # src
                maximum,  # threshold value
                0,  # we dont care because of the selected type
                cv2.THRESH_TOZERO_INV  # t type
            )

            (t, self.channels[channel]) = cv2.threshold(
                tmp,  # src
                minimum,  # threshold value
                255,  # maxvalue
                cv2.THRESH_BINARY  # type
            )

            #if channel == 'hue':
                # only works for filtering red color because the range for the hue
                # is split
             #   self.channels['hue'] = cv2.bitwise_not(self.channels['hue'])
                #self.channels['hue'] = cv2.bitwise_and(self.channels['hue'], self.channels['hue_blu'])

        def track(self, frame, mask):
            """
            Track the position of the laser pointer.
            Code taken from
            http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

            center = None

            countours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)[-2]

            # only proceed if at least one contour was found
            if len(countours) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(countours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                moments = cv2.moments(c)
                if moments["m00"] > 0:
                    center = int(moments["m10"] / moments["m00"]), \
                             int(moments["m01"] / moments["m00"])
                else:
                    center = int(x), int(y)

                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    # then update the pointer trail
                    if self.previous_position:
                        cv2.line(self.trail, self.previous_position, center,
                                 (255, 255, 255), 2)

            cv2.add(self.trail, frame, frame)
            self.previous_position = center
            """
            # find contours in the mask and initialize the current
            # (x, y) center of the ball
            pts = deque(maxlen=64)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None
            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)

                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = numpy.int0(box)
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                #Find the object rotationa angle, its center and finds the line going throught the center and parallel to the object countor (its axes)
                angle = rect[2]
                m_line = numpy.tan(angle * numpy.pi / 180)
                x_center = rect[0][0]
                y_center = rect[0][1]
                visible_lenght = rect[1][0]
                q_line = y_center - x_center * m_line
                x_point = 1280 #Substitute with self.cam_width
                y_point = int(x_point * m_line + q_line)
                #cv2.line(frame, (int(x_center), int(y_center)), (x_point, y_point), (0, 255, 0), 2)
                #Now knowing the the axes, the real object lenght and tis center we can find where the real edge is
                    #First I caluclate half the side of the contour
                half_side = visible_lenght/2 #(numpy.sqrt(numpy.square(box[0][0] - box[3][0]) + numpy.square(box[0][1] - box[3][1])))/2
                    #Than I find the radius of the points with center in the countr center and distant le object lenght - half the dise
                radius = OBJECT_LENGHT - half_side #+ numpy.sqrt(numpy.square(y_center) + numpy.square(x_center))
                    #Knowing the line angle i can find the points of the edge with cos and sin functions
                x_edge = radius * numpy.cos(angle * numpy.pi /180) + x_center
                y_edge = radius * numpy.sin(angle * numpy.pi /180) + y_center
                """print(radius)
                print(x_edge)
                print(y_edge)
                cv2.circle(frame, (int(x_edge), int(y_edge)), 50,
                           (0, 255, 255), 2)

                print(int(x_center))
                print(int(y_center))
                print(x_point)
                print(y_point)
                print("Bordi scatola:")
                print(box)
                print("\nCentro, lunghezza e larghezza, angolo di rotazione: ")
                print(rect) """

                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                # M>0 to avoid ZeroDivisionError
                if M["m00"] > 0:
                    center = int(M["m10"] / M["m00"]), \
                             int(M["m01"] / M["m00"])
                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    #cv2.circle(frame, center, 5, (0, 0, 255), -1)
            # update the points queue
            pts.appendleft(center)

            # loop over the set of tracked points
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(numpy.sqrt(64 / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        def detect(self, frame):
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


            # split the video frame into color channels
            h, s, v = cv2.split(hsv_img)


            self.channels['hue'] = h
            self.channels['saturation'] = s
            self.channels['value'] = v


            self.values()
            # Threshold ranges of HSV components; storing the results in place
            self.threshold_image("hue")
            self.threshold_image("saturation")
            self.threshold_image("value")

            # Perform an AND on HSV components to identify the laser!
            self.channels['laser'] = cv2.bitwise_and(
                self.channels['hue'],
                self.channels['value']
            )
            self.channels['laser'] = cv2.bitwise_and(
                self.channels['saturation'],
                self.channels['laser']
            )

            # Merge the HSV components back together.
            hsv_image = cv2.merge([
                self.channels['hue'],
                self.channels['saturation'],
                self.channels['value'],
            ])

            # A series of erosions and dilations to remove any small blobs that may be left on the mask.
            #self.channels['laser'] = cv2.erode(self.channels['laser'], None, iterations=2)
            #self.channels['laser'] = cv2.dilate(self.channels['laser'], None, iterations=2)

            self.track(frame, self.channels['laser'])

            return hsv_image

        def display(self, img, frame):
            """Display the combined image and (optionally) all other image channels
            NOTE: default color space in OpenCV is BGR.
            """
            cv2.imshow('RGB_VideoFrame', frame)
            cv2.imshow('LaserPointer', self.channels['laser'])
            if self.display_thresholds:
                cv2.imshow('Thresholded_HSV_Image', img)
                cv2.imshow('Hue', self.channels['hue'])
                cv2.imshow('Saturation', self.channels['saturation'])
                cv2.imshow('Value', self.channels['value'])

        def setup_windows(self):
            sys.stdout.write("Using OpenCV version: {0}\n".format(cv2.__version__))

            # create output windows
            self.create_and_position_window('LaserPointer', 0, 0)
            self.create_and_position_window('RGB_VideoFrame',
                                            10 + self.cam_width, 0)
            if self.display_thresholds:
                self.create_and_position_window('Thresholded_HSV_Image', 10, 10)
                self.create_and_position_window('Hue', 20, 20)
                self.create_and_position_window('Saturation', 30, 30)
                self.create_and_position_window('Value', 40, 40)

        def run(self):
            # Set up window positions
            self.setup_windows()
            # Set up the camera capture
            self.setup_camera_capture()

            while True:
                # 1. capture the current image
                #success, frame = self.capture.read() DECOMMENTARE PER USARE LA WEBCAM LINEE 295-299
                frame  = cv2.imread("Foto_prova_contorni.jpg")
                #if not success:  # no image captured... end the processing
                #    sys.stderr.write("Could not read camera frame. Quitting\n")
                #    sys.exit(1)

                hsv_image = self.detect(frame)
                self.display(hsv_image, frame)
                self.handle_quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Laser Tracker')
    parser.add_argument('-W', '--width',
                        default=640,
                        type=int,
                        help='Camera Width')
    parser.add_argument('-H', '--height',
                        default=480,
                        type=int,
                        help='Camera Height')
    parser.add_argument('-u', '--huemin',
                        default=20,
                        type=int,
                        help='Hue Minimum Threshold')
    parser.add_argument('-U', '--huemax',
                        default=160,
                        type=int,
                        help='Hue Maximum Threshold')
    parser.add_argument('-s', '--satmin',
                        default=100,
                        type=int,
                        help='Saturation Minimum Threshold')
    parser.add_argument('-S', '--satmax',
                        default=255,
                        type=int,
                        help='Saturation Maximum Threshold')
    parser.add_argument('-v', '--valmin',
                        default=200,
                        type=int,
                        help='Value Minimum Threshold')
    parser.add_argument('-V', '--valmax',
                        default=255,
                        type=int,
                        help='Value Maximum Threshold')
    parser.add_argument('-d', '--display',
                        action='store_true',
                        help='Display Threshold Windows')
    params = parser.parse_args()

    tracker = LaserTracker(
        cam_width=params.width,
        cam_height=params.height,
        hue_min=params.huemin,
        hue_max=params.huemax,
        sat_min=params.satmin,
        sat_max=params.satmax,
        val_min=params.valmin,
        val_max=params.valmax,
        display_thresholds=params.display
    )
    tracker.run()