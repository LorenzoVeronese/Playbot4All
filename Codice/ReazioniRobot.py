import RPi.GPIO as GPIO
import time

class ReazioniRobot:
    braccio_sx = 0
    braccio_dx = 0
    orecchie = 0


    def __init__(self, pin_braccio_sx, pin_braccio_dx, pin_orecchie):
        GPIO.setmode(GPIO.BCM)

        GPIO.setup(pin_braccio_sx, GPIO.OUT)
        self.braccio_sx = GPIO.PWM(pin_braccio_sx, 50)#Antiorario = alto, orario = basso
        GPIO.setup(pin_braccio_dx, GPIO.OUT)
        self.braccio_dx = GPIO.PWM(pin_braccio_dx, 50)#Orario = alto, antiorario = basso
        GPIO.setup(pin_orecchie, GPIO.OUT)
        self.orecchie = GPIO.PWM(pin_orecchie, 50)#Antiorario = basso, orario = alto



    def reazionePostitiva(self):
        #Alzo le braccia di 90 gradi e le orecchie di quasi 180 gradi
        self.braccio_sx.start(0) #Inizializzo le braccia
        self.braccio_dx.start(0)

        self.braccio_sx.ChangeDutyCicle(15-5) #Gira di 90 gradi verso l'alto
        self.braccio_dx.ChangeDutyCicle(5) #Gira di 90 gradi verso l'alto
        time.sleep(1)

        self.braccio_sx.stop()
        self.braccio_dx.stop()

        self.orecchie.start(0)
        self.orecchie.ChangeDutyCicle(11)
        time.sleep(1)

        self.orecchie.stop()

    def reazioneNegativad(self):
        # Alzo le braccia di 90 gradi e le orecchie di quasi 180 gradi
        self.braccio_sx.start(0)  # Inizializzo le braccia
        self.braccio_dx.start(0)

        self.braccio_sx.ChangeDutyCicle(5)  # Gira di 90 gradi verso il basso
        self.braccio_dx.ChangeDutyCicle(15)  # Gira di 90 gradi verso il basso
        time.sleep(1)

        self.braccio_sx.stop()
        self.braccio_dx.stop()

        self.orecchie.start(0)
        self.orecchie.ChangeDutyCicle(4)
        time.sleep(1)

        self.orecchie.stop()
