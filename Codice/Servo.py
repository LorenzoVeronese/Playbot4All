import RPi.GPIO as GPIO
import time

# set pinmode
GPIO.setmode(GPIO.BCM)

# set servo's pin
GPIO.setup(18, GPIO.OUT)
# set frequency (50Hz = 20ms)
arm_sx = GPIO.PWM(18, 50)
# set initial value of the signal
arm_sx.start(0)

# this will be modified according to what we'll do
arm_sx.ChangeDutyCycle(2) #max speed clockwise
time.sleep(5) # gives time to move
arm_sx.ChangeDutyCycle(2) #max speed anti-clockwise

# finally
arm_sx.stop()
GPIO.cleanup()