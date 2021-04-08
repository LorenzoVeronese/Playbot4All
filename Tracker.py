import cv2
import numpy
import sys
import statistics
import imutils


class Tracker(object):
    def __init__(self, LASER, HAND, HAND_MASK):
        self.camera = None
        self.frame = None
        self.prev_frame = None
        self.laser_pos = (0, 0)
        self.hand_pos = (0, 0)
        self.hand_mask = None
        # what to display (debugging): see funct 'display'
        self.display_flags = {
            'laser' : LASER, 
            'hand' : HAND, 
            'hand_mask' : HAND_MASK
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


    def laser_tracking(self):
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

        # NOTE: catch error when you have white light
        M = cv2.moments(self.hand_mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

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
            circle = cv2.circle(frame, self.laser_pos, 100, (255, 0, 0), 4)

        if self.display_flags['hand'] == 1: # HAND on image
            # gree circle for hand
            circle = cv2.circle(frame, self.hand_pos, 200, (0, 255, 0), 4)

        if self.display_flags['hand_mask'] == 1: # HAND_MASK
            to_display['hand_mask'] = self.hand_mask
        
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
    tracker = Tracker(LASER=1, HAND=1, HAND_MASK=1)
    tracker.run()
    cv2.destroyAllWindows()