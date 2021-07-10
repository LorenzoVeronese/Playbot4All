import RPi.GPIO as GPIO
import time



class laser:
    laserA1 = 0;
    
    def __init__(self, a1):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        self.laserA1 = a1
        GPIO.setup(self.laserA1, GPIO.OUT)
        
    def laser_on(self):
        print("led on")
        GPIO.output(self.laserA1, 1)
    
    def laser_off(self):
        print("led off")
        GPIO.output(self.laserA1, 0)        