import cv2
import numpy
import sys
import statistics
import imutils


class Tracker(object):
    def __init__(self, LASER, HAND, HAND_MASK, PAPER_MASK):
        self.camera = None
        self.frame = None
        self.prev_frame = None
        self.laser_pos = (0, 0)
        self.hand_pos = (0, 0)
        self.hand_mask = None
        self.paper_mask = None
        # what to display (debugging): see funct 'display'
        self.display_flags = {
            'laser' : LASER, 
            'hand' : HAND, 
            'hand_mask' : HAND_MASK,
            'paper_mask' : PAPER_MASK
        }


    def setup_camera_capture(self, device_num = 0):
        """
        Perform camera setup for the device number (default device = 0).
        Return a reference to the camera Capture object.
        """
        try:
            device = int(device_num)
            sys.stdout.write(f'Using Camera Device: {device}\n')
        except (IndexError, ValueError):
            # set default device
            device = 0
            sys.stderr.write("Invalid Device. Using default device 0\n")

        # Try to start capturing frames
        self.camera = cv2.VideoCapture(device, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            sys.stderr.write("Failed to Open Capture device. Quitting.\n")
            sys.exit(1)

        return self.camera


    def paper_tracking(self):
        """
        find the paper. This is useful to select the input only in the interesting
        area.
        EXECUTION
        mask of bgr and hsv
        bitwise and between them
        """
        frame = self.frame.copy()

        lower = numpy.array([85, 85, 90], dtype="uint8")  # 0, 48, 80
        upper = numpy.array([255, 255, 255], dtype="uint8")  # 20, 255, 255
        color_mask = cv2.inRange(frame, lower, upper)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = numpy.array([0, 0, 120], dtype="uint8")  # 105
        upper = numpy.array([179, 110, 255], dtype="uint8")  # 20, 255, 255
        hsv_mask = cv2.inRange(hsv, lower, upper)

        self.paper_mask = cv2.bitwise_and(hsv_mask, color_mask)

        blank = numpy.zeros((len(frame), len(frame[0])), dtype='uint8')
        paper, thresh = cv2.threshold(self.paper_mask, 40, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            # draw in blue the contours that were founded
            cv2.drawContours(blank, contours, -1, (255, 255, 0), -1)
        
        self.paper_mask = blank
        return self.paper_mask


    # NOTE: this doesn't work so well: see laser_tracking
    def laser_tracking_old(self):
        """
        Find the laser
        EXECUTION
        this splits color channels: I take red, I find the position
        of the pixel with max red value (red laser)
        NOTE: the red light in the hand distracts this tracker, which
            start to go on the hand and not on the laser. So I decided
            to use also brightness (see when I use h) and then take the mean
            as the center value
        """
        frame = self.frame.copy()
        b, g, r = cv2.split(frame)

        # finds max's coordinates of the red channel
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

        # same as for red: find the max coordinates of the h channel
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(frame)
        bright_index = numpy.unravel_index(numpy.argmax(h, axis=None), h.shape)

        # put together
        rows = [laser_index[1], bright_index[1]]
        cols = [laser_index[0], bright_index[0]]

        # compute the mean value
        self.laser_pos = (int(statistics.mean(rows)), int(statistics.mean(cols)))
        return self.laser_pos


    def laser_tracking(self):
        """
        Find the laser 
        EXECUTION
        set lower and upper value of threshold, find moment (center of all
        regions found): this is (more or less) where the laser is positioned
        """
        frame = self.frame

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = numpy.array([0, 0, 230], dtype="uint8")  # 0, 48, 80
        upper = numpy.array([20, 100, 255], dtype="uint8")  # 20, 255, 255
        laser_mask = cv2.inRange(hsv, lower, upper)

        M = cv2.moments(laser_mask)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except (ZeroDivisionError):
            cX = 0
            cY = 0

        self.laser_pos = (cX - 25, cY - 25)
        return self.laser_pos
        

    def hand_tracking(self):
        """
        Find the hand
        EXECUTION
        set lower and upper value of threshold, find moment (center of all
        regions found): this is where the hand is positioned
        """
        hsv = cv2.cvtColor(self.frame.copy(), cv2.COLOR_BGR2HSV)
        lower = numpy.array([0, 80, 100], dtype="uint8")  # 0, 48, 80
        upper = numpy.array([20, 200, 200], dtype="uint8")  # 20, 255, 255
        self.hand_mask = cv2.inRange(hsv, lower, upper)
        #hand_mask_and = cv2.bitwise_and(self.hand_mask, self.paper_mask)
        self.hand_mask = cv2.bitwise_and(self.hand_mask, self.paper_mask)


        # NOTE: catch error when you have white light
        M = cv2.moments(self.hand_mask)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except (ZeroDivisionError):
            cX = 0
            cY = 0

        self.hand_pos = (cX - 25, cY - 25)
        return self.hand_pos


    def display(self):
        """
        Display video for debugging according to flags
        EXECUTION
        create a dictionary in which is put any video required
        """
        frame = self.frame.copy()
        to_display = {'frame_circles':frame, 'hand_mask':None}

        if self.display_flags['laser'] == 1: # LASER on image
            # red circle for laser
            circle = cv2.circle(frame, self.laser_pos, 50, (255, 0, 0), 4)

        if self.display_flags['hand'] == 1: # HAND on image
            # gree circle for hand
            circle = cv2.circle(frame, self.hand_pos, 150, (0, 255, 0), 4)

        if self.display_flags['hand_mask'] == 1: # HAND_MASK
            to_display['hand_mask'] = self.hand_mask

        if self.display_flags['paper_mask'] == 1: # PAPER_MASK
            to_display['paper_mask'] = self.paper_mask
        
        return to_display


    def run(self, camera_num = 0):
        """
        Run the program
        """
        # Set up the camera capture
        self.setup_camera_capture(camera_num)
        
        while True:
            # capture the current image
            check, self.frame = self.camera.read()
            if not check:  # no image captured... end the processing
                sys.stderr.write("Could not read camera frame. Quitting\n")
                sys.exit(1)
            
            self.paper_tracking()
            self.laser_tracking()
            self.hand_tracking()

            # display videos according to flags set
            to_display = self.display()
            for video in to_display.items():
                # TODO: catch the case in which the video is not in the
                # dictionary (flag off)
                cv2.imshow(f'{video[0]}', video[1])
            
            key = cv2.waitKey(1)
            if key == 27:
                break
        
        self.camera.release()


if __name__ == '__main__':
    tracker = Tracker(LASER=1, HAND=1, HAND_MASK=1, PAPER_MASK=1)
    tracker.run()
    cv2.destroyAllWindows()
