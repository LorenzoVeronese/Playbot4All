import cv2
import numpy
import argparse
import sys

class LaserTracker(object):

    def __init__(self, cam_width=640, cam_height=480, hue_min=20, hue_max=160,
             sat_min=100, sat_max=255, val_min=200, val_max=256,
             display_thresholds=True):

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
        """Creates a named window placing it on the screen at (xpos, ypos)."""
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
            # set device number
            device = int(device_num)
            # writing on terminal the device's number used
            sys.stdout.write("Using Camera Device: {0}\n".format(device))
        except (IndexError, ValueError):
            # assume we want the 1st device
            device = 0
            sys.stderr.write("Invalid Device. Using default device 0\n")
        # Try to start capturing frames
        self.capture = cv2.VideoCapture(device)
        if not self.capture.isOpened():
            sys.stderr.write("Faled to Open Capture device. Quitting.\n")
            sys.exit(1)
        # set the wanted image size from the camera
        self.capture.set(
            # if the opencv version starts with 2 use the first one, otherwise the other one (I think that
            # in the newer version of opencv you need to write cv2.cv....)
            cv2.cv.CV_CAP_PROP_FRAME_WIDTH if cv2.__version__.startswith('2') else cv2.CAP_PROP_FRAME_WIDTH,
            self.cam_width
        )
        self.capture.set(
            # same as above
            cv2.cv.CV_CAP_PROP_FRAME_HEIGHT if cv2.__version__.startswith('2') else cv2.CAP_PROP_FRAME_HEIGHT,
            self.cam_height
        )
        return self.capture

    def handle_quit(self, delay=10):
        """Quit the program if the user presses "Esc" or "q"."""
        # opencv wait until the user enter a key
        key = cv2.waitKey(delay)
        # unicode character of the key
        c = chr(key & 255)
        # pressing 'c' resets the white line
        if c in ['c', 'C']:
            # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
            self.trail = numpy.zeros((self.cam_height, self.cam_width, 3),
                                     numpy.uint8)
        # pressing 'q' or 'esc' terminates the program
        if c in ['q', 'Q', chr(27)]:
            sys.exit(0)

    # threshold means that a value upon a limit is one, under is zero
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
        if channel == 'hue':
            # only works for filtering red color because the range for the hue
            # is split
            # this exchanges 1s and 0s
            self.channels['hue'] = cv2.bitwise_not(self.channels['hue'])

    def track(self, frame, mask):
        """
        Track the position of the laser pointer.
        Code taken from
        http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
        """
        center = None
        # https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html
        # RETR_EXTERNAL: If you use this flag, it returns only extreme outer flags. 
        # All child contours are left behind. We can say, under this law, Only 
        # the eldest in every family is taken care of. It doesn't care about other ù
        # members of the family :).
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
        # cv2.CHAIN_APPROX_SIMPLE: If you pass cv2.CHAIN_APPROX_NONE, all the boundary points are stored.
        # But actually do we need all the points? For eg, you found the contour of a 
        # straight line. Do you need all the points on the line to represent that line? 
        # No, we need just two end points of that line. This is what cv2.CHAIN_APPROX_SIMPLE 
        # does. It removes all redundant points and compresses the contour, thereby 
        # saving memory.
        # ?why the '[-2]'?
        countours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]
        # only proceed if at least one contour was found
        if len(countours) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = min(countours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            moments = cv2.moments(c)
            # 3 moments calculated (<= up to 3rd order): 00, 01, 10
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
                # then update the ponter trail
                if self.previous_position:
                    cv2.line(self.trail, self.previous_position, center,
                             (255, 255, 255), 2)
        # to display the line
        cv2.add(self.trail, frame, frame)
        self.previous_position = center

    def detect(self, frame):
        # colours to gray conversion
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # split the video frame into color channels
        h, s, v = cv2.split(hsv_img)
        self.channels['hue'] = h
        self.channels['saturation'] = s
        self.channels['value'] = v
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
            success, frame = self.capture.read()
            if not success:  # no image captured... end the processing
                sys.stderr.write("Could not read camera frame. Quitting\n")
                sys.exit(1)
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