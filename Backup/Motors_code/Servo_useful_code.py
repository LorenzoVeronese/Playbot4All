import RPi.GPIO as GPIO
import time

'''
def SetAngle(angle, servo_pin):
    """
    angle: how much I want to rotate
    pin: on which pin I want to act
    """
    duty = angle/18 + 2
    GPIO.output(servo_pin, True)
    arm_sx.ChangeDutyCycle(duty)
    time.sleep(10)
    GPIO.output(servo_pin, False)
    arm_sx.ChangeDutyCycle(0)
'''

GPIO.setmode(GPIO.BCM)

GPIO.setup(18, GPIO.OUT)
arm_sx = GPIO.PWM(18, 50)

arm_sx.start(1.5)
print("1 Wainting for 1 seconds")
time.sleep(1)
'''
SetAngle(90, 18)

arm_sx.stop()
GPIO.cleanup()
'''
print("Rotating 180 degrees in 10 steps")

arm_sx.ChangeDutyCycle(1.2)
'''
duty = 2

while duty <= 12:
    arm_sx.ChangeDutyCycle(duty)
    time.sleep(4)
    arm_sx.ChangeDutyCycle(0)
    time.sleep(2)
    duty += 1
'''
print("2 Wainting for 2 seconds")
time.sleep(5)

print("Back to 90 degrees for 2 seconds")
arm_sx.ChangeDutyCycle(1.8)
print("3 Wainting for 2 seconds")
time.sleep(2)

print("Turning back to 0 degrees")
arm_sx.ChangeDutyCycle(1.5)

print("Finished")
time.sleep(2)

arm_sx.stop()
GPIO.cleanup()

'''
from gpiozero import Servo
from time import sleep

servo = Servo(18)

while True:
    servo.min()
    sleep(5)
    servo.mid()
    sleep(5)
    servo.max()
    sleep(5)
    break'''