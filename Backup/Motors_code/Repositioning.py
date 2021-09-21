import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
'''
#Asse x
pin1 = 17
pin2 = 12 #changed 18 to 12
pin3 = 21
pin4 = 22
'''
#Asse y
pin1 = 23
pin2 = 24
pin3 = 25
pin4 = 27

GPIO.setup(pin1, GPIO.OUT)
GPIO.setup(pin2, GPIO.OUT)
GPIO.setup(pin3, GPIO.OUT)
GPIO.setup(pin4, GPIO.OUT)

def forward(delay, steps):
    for i in range(0, steps):
        setStep(1, 1,0,0)
        #time.sleep(delay)
        setStep(0, 1,0,0)
        time.sleep(delay)
        setStep(0,1,1,0)
        #time.sleep(delay)
        setStep(0,0,1,0)
        time.sleep(delay)
        setStep(0,0,1,1)
        #time.sleep(delay)
        setStep(0,0,0,1)
        time.sleep(delay)
        setStep(1,0,0,1)
        #time.sleep(delay)
        setStep(1,0,0,0)
        time.sleep(delay)
        
def backward(delay, steps):
    for i in range(0, steps):
        setStep(1,0,0,0)
        time.sleep(delay)
        setStep(0,0,0,1)
        time.sleep(delay)
        setStep(0,0,1,0)
        time.sleep(delay)
        setStep(0, 1,0,0)
        time.sleep(delay)
        
        
        
        
def setStep(w1, w2, w3, w4):  
    GPIO.output(pin1, w1)
    GPIO.output(pin2, w2)
    GPIO.output(pin3, w3)
    GPIO.output(pin4, w4)

while True:
    #dealy = raw_input("Delay:")
    #steps = raw_input("Steps:")
    #forward(3 / 1000.0, 512)
    backward(3 / 1000.0, 512)