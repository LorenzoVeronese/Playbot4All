import cv2
import numpy
import sys
import statistics
import imutils
from collections import deque


class Tracker(object):
    def __init__(self, LASER, HAND, PAPER_MASK, LASER_MASK, HAND_MASK, PEN, PEN_MASK, TIP_MASK):
        # camera
        self.camera = None
        self.frame = None
        self.prev_frame = None # actually not used
        # laser and hand positions
        self.laser_pos = (0, 0)
        self.hand_pos = (0, 0)
        # laser and hand masks
        self.paper_mask = None
        self.laser_mask = None
        self.hand_mask = None
        # pen
        self.pen_box = None
        self.pen_line = (0, 0, 0, 0)
        self.pen_tip_pos = (0, 0)
        self.pen_mask = None
        self.tip_mask = None
        # what to display (debugging): see funct 'display'
        self.display_flags = {
            'laser' : LASER, # red circle for the laser
            'hand' : HAND, # green circle the hand
            'laser_mask' : LASER_MASK, # threshold mask of the laser
            'paper_mask' : PAPER_MASK, # threshold mask of the paper (see paper_setup)
            'hand_mask' : HAND_MASK, # threshold mask of the hand
            'pen' : PEN, # blue box for the pen, light blue dot for the pen tip
            'pen_mask' : PEN_MASK, # threshold mask of the pen
            'tip_mask' : TIP_MASK # to see intersection between hand circle and pen line
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


    def setup_paper_mask(self):
        """
        When you start the program, you need to know where the paper is:
        this create the mask of it.
        NOTE 1: there must be nothing on the paper! The paper must be the only
        "relevant" object on the desk
        NOTE 2: this stage is in the very beginning of the run, so the camera
        must be directed to the paper from the start
        NOTE 3: if the camera will move with the head, we will need to
        move also the hand_mask created at this step
        """
        i = 0
        for i in range(0, 30):
            check, self.frame = self.camera.read()
            self.paper_tracking()
        self.camera.release()


    def paper_tracking(self):
        """
        find the paper. This is useful to select the input only in the interesting
        area.
        EXECUTION
        mask of bgr and hsv
        bitwise and between them
        """
        frame = self.frame.copy()

        # define a bgr based mask
        lower = numpy.array([85, 85, 90], dtype="uint8")  # 0, 48, 80
        upper = numpy.array([255, 255, 255], dtype="uint8")  # 20, 255, 255
        color_mask = cv2.inRange(frame, lower, upper)

        # define a hsv based mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = numpy.array([0, 0, 120], dtype="uint8")  # 105
        upper = numpy.array([179, 110, 255], dtype="uint8")  # 20, 255, 255
        hsv_mask = cv2.inRange(hsv, lower, upper)

        # and between the two masks: more precision
        self.paper_mask = cv2.bitwise_and(hsv_mask, color_mask)

        # find paper contour
        blank = numpy.zeros((len(frame), len(frame[0])), dtype='uint8')
        paper, thresh = cv2.threshold(self.paper_mask, 40, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # fill of white the paper contour found
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            # draw in blue the contours that were founded
            cv2.drawContours(blank, c, -1, 255, -1)
            # fill the area inside the contours founded
            cv2.fillPoly(blank, pts=[c], color=(255, 255, 255))
        
        self.paper_mask = blank
        return self.paper_mask


    # NOTE: this is very good when there's no red light which reflects
    # on the hand
    def laser_tracking_1(self):
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

        # "fake" laser_mask. This laser_mask is useful only with laser_tracking_2:
        # for laser_tracking_1 I put this fake one to switch easily from one
        # function to the other
        blank = numpy.zeros((len(frame), len(frame[0])), dtype='uint8')
        self.laser_mask = cv2.circle(blank, self.laser_pos, 10, (255, 0, 0), -1)

        return self.laser_pos


    # NOTE: this is not very good when there is a small red dot and no
    # red reflects on there surfaces. In general is better laser_tracking_1
    def laser_tracking_2(self): # unused: see laser_tracking_1 for the current version
        """
        Find the laser 
        EXECUTION
        set lower and upper value of threshold, find moment (center of all
        regions found): this is (more or less) where the laser is positioned
        """
        frame = self.frame

        # define hsv based mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = numpy.array([0, 0, 200], dtype="uint8")  # 0, 48, 80
        upper = numpy.array([40, 255, 255], dtype="uint8")  # 20, 255, 255
        self.laser_mask = cv2.inRange(hsv, lower, upper)

        # find center (moment) of the area found
        M = cv2.moments(self.laser_mask)
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
        # define and find hsv based mask
        hsv = cv2.cvtColor(self.frame.copy(), cv2.COLOR_BGR2HSV)
        lower = numpy.array([0, 80, 100], dtype="uint8")  # 0, 48, 80
        upper = numpy.array([20, 200, 200], dtype="uint8")  # 20, 255, 255
        self.hand_mask = cv2.inRange(hsv, lower, upper)
        
        # I don't want things which aren't on the paper
        self.hand_mask = cv2.bitwise_and(self.hand_mask, self.paper_mask)

        # find center of the area found
        # TODO: catch error when you have white light using last not zero value
        M = cv2.moments(self.hand_mask)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except (ZeroDivisionError):
            cX = 0
            cY = 0

        self.hand_pos = (cX - 25, cY - 25)
        return self.hand_pos


    def pen_tracking(self):
        """
        find the pen tip
        EXECUTION
        find contour of the pen using a mask.
        using a box surrounding the contour, find the pen axis
        intersect hand circle and pen line to find the pen tip (approximately)
        """
        # PEN SEARCHING
        # define and find hsv based mask
        hsv = cv2.cvtColor(self.frame.copy(), cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lower = numpy.array([17, 95, 111], dtype = "uint8")
        upper = numpy.array([65, 255, 255], dtype = "uint8")
        self.pen_mask = cv2.inRange(hsv, lower, upper)

        # find contours
        pts = deque(maxlen=64)
        cnts = cv2.findContours(self.pen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        center = None
        # I need at least one contour
        if len(cnts) > 0: 
            # find largest contour
            c = max(cnts, key=cv2.contourArea)

            # put a rectangle which surround the contour (pen) found
            rect = cv2.minAreaRect(c)
            self.pen_box = cv2.boxPoints(rect)
            self.pen_box = numpy.int0(self.pen_box)

            # Find the object rotational angle and its center, then finds the 
            # line going throught the center and parallel to the object 
            # countor (its axes)
            if self.pen_box[0][1] > self.pen_box[2][1]:
                highest_point_2 = self.pen_box[0]
            else:
                highest_point_2 = self.pen_box[2]
            lowest_point = self.pen_box[1]

            # set a line along the pen
            m_line = (highest_point_2[1] - lowest_point[1]) / (highest_point_2[0] - lowest_point[0])
            x_center = rect[0][0]
            y_center = rect[0][1]
            visible_lenght = rect[1][0]
            q_line = y_center - x_center * m_line
            x_point = len(self.frame[0])  # the line must "cut" the frame

            # draw the pen-line
            try:
                y_point = int(x_point * m_line + q_line)
                x_center = int(x_center)
                y_center = int(y_center)
                # save some points for display purpose (see funct self.display())
                self.pen_line = (x_center, y_center, x_point, y_point)
            except:
                # TODO: catch when the pen is not in sight
                y_point = 1
                print("Cannot see pen")

            # TIP PEN SEARCHING
            # set on a blank the hand circle and the pen line
            blank_line = numpy.zeros((len(self.frame), len(self.frame[0])), dtype='uint8')
            blank_circle = blank_line
            cv2.line(blank_line, (int(x_center), int(y_center)),(x_point, y_point), (255, 255, 255), 10)
            cv2.circle(blank_circle, self.hand_pos, 150, (255, 255, 255), 10)
               
            # intersection between line and circle
            tip_mask = cv2.bitwise_and(blank_line, blank_circle)
            self.tip_mask = tip_mask
            cv2.circle(self.tip_mask, self.hand_pos, 150, (0, 255, 0), -1)

            # only one point remained
            M = cv2.moments(tip_mask)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except (ZeroDivisionError):
                cX = 0
                cY = 0
            self.pen_tip_pos = (cX - 25, cY - 25)


    def display(self):
        """
        Display video for debugging according to self.display_flags
        EXECUTION
        create a dictionary in which any video required is put
        """
        frame = self.frame.copy()
        # show the video by default. On it we can draw signals (see lines below)
        to_display = {'frame' : frame}

        if self.display_flags['laser'] == 1: # LASER on image
            # red circle for laser
            circle = cv2.circle(frame, self.laser_pos, 30, (0, 0, 255), 3)

        if self.display_flags['hand'] == 1: # HAND on image
            # green circle for hand
            circle = cv2.circle(frame, self.hand_pos, 120, (0, 255, 0), 3)

        if self.display_flags['paper_mask'] == 1: # PAPER_MASK
            to_display['paper_mask'] = self.paper_mask

        if self.display_flags['laser_mask'] == 1: # LASER_MASK
            to_display['laser_mask'] = self.laser_mask
 
        if self.display_flags['hand_mask'] == 1: # HAND_MASK
            to_display['hand_mask'] = self.hand_mask

        if self.display_flags['pen'] == 1: # PEN (box which contains it)
            # blue box on the pen
            cv2.drawContours(frame, [self.pen_box], 0, (255, 0, 0), 3)
            # yellow line along the pen
            cv2.line(frame, (self.pen_line[0], self.pen_line[1]), (self.pen_line[2], self.pen_line[3]), (0, 255, 255), 3)
            # light blue on the pen tip
            cv2.circle(frame, self.pen_tip_pos, 10, (255, 255, 0), -1)
         
        if self.display_flags['pen_mask'] == 1: # PEN_MASK to check the color
            to_display['pen_mask'] = self.pen_mask

        if self.display_flags['tip_mask'] == 1: # TIP_MASK to check the intersection hand circle - pen line
            to_display['tip_mask'] = self.tip_mask
        
        return to_display


    def run(self, camera_num = 0):
        """
        Run the program
        """
        # Set up the camera capture and paper_mask
        self.setup_camera_capture(camera_num)
        self.setup_paper_mask()
        self.setup_camera_capture(camera_num)

        while True:
            # capture the current image
            check, self.frame = self.camera.read()
            if not check:  # no image captured... end the processing
                sys.stderr.write("Could not read camera frame. Quitting\n")
                sys.exit(1)
            
            # make trackings
            # self.paper_tracking() this is not used here anymore: see paper_mask_setup
            self.laser_tracking_1()
            self.hand_tracking()
            self.pen_tracking()

            # display videos according to flags set
            to_display = self.display()
            if len(to_display.items()) == 0:
                pass
            else:
                for video in to_display.items():
                    if type(video[1]) == type(None):
                        pass
                    else:
                        print(type(video[1]))
                        cv2.imshow(f'{video[0]}', video[1])
            
            # to stop the process
            key = cv2.waitKey(1)
            if key == 27:
                break
        
        self.camera.release()


if __name__ == '__main__':
    tracker = Tracker(LASER=1, HAND=1, PAPER_MASK=0, LASER_MASK=0, HAND_MASK=0, PEN = 1, PEN_MASK=0, TIP_MASK=0)

    tracker.run()
    
    cv2.destroyAllWindows()