import RPi.GPIO as GPIO
import time
from gpiozero import LED

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
'''
led = LED(26)
led.on()
time.sleep(100)
led.off()

'''
GPIO.setup(26, GPIO.OUT)

print("led on")
GPIO.output(26, 1)
time.sleep(10)

print("led off")
GPIO.output(26, 0)
time.sleep(100)

GPIO.cleanup()