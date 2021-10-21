import RPi.GPIO as GPIO
from stepper_motor_control import stepper_motor
import os
import time

class Position_Control():
    def __init__(self, motor_x, motor_y, delay_x, delay_y):
        self.posizione = 1
        self.x_stepper = motor_x
        self.y_stepper = motor_y
        self.x_delay = delay_x
        self.y_delay = delay_y
        
    def save_position(self, x, y): #salva la posizione della struttura
        if self.posizione == 2:
            file = open("/home/pi/Desktop/Playbot4All-main/Motors_code/positions1.txt", "w")
            file.write(str(x) + "," + str(y) + ",")
            file.close()
            self.posizione = 1
        else:
            file = open("/home/pi/Desktop/Playbot4All-main/Motors_code/positions2.txt", "w")
            file.write(str(x) + "," + str(y) + ",")
            file.close()
            self.posizione = 2

        file = open("/home/pi/Desktop/Playbot4All-main/Motors_code/valore_posizione.txt", "w")
        file.write(str(self.posizione))
        file.close()
        
    def sign(self, n): #return the sign of a number
      if n > 0:
        return 1
      elif n < 0:
        return -1
      else:
        return 0

    def move_and_save(self,x_distance, y_distance, x, y):
        x_direction = self.sign(x_distance)
        y_direction = self.sign(y_distance)
        x_no_steps = abs(x_distance)
        y_no_steps = abs(y_distance)
        
        print( "X no steps: " + str(x_no_steps))
        print ("Y no steps: " + str(y_no_steps))
        
        over = 0
        if (x_no_steps > y_no_steps):
            for i in range(0, x_no_steps):
                if (x_direction > 0):
                    x += 1
                elif (x_direction < 0):
                    x -= 1
                self.x_stepper.motor_move(x, self.x_delay)
                over += y_no_steps
                if (over >= x_no_steps):
                    if (y_direction > 0):
                        y += 1
                    elif (y_direction < 0):
                        y -= 1
                    self.y_stepper.motor_move(y, self.y_delay)
                    over -= x_no_steps
                self.posizione = self.save_position(x,y)
        else:
            for i in range(0, y_no_steps):
                if (y_direction > 0):
                    y += 1
                elif (y_direction < 0):
                    y -= 1
                self.y_stepper.motor_move(y, self.y_delay)
                over += x_no_steps
                if (over >= y_no_steps):
                    if (x_direction > 0):
                        x += 1
                    elif (x_direction < 0):
                        x -= 1
                    self.x_stepper.motor_move(x, self.x_delay)
                    over -= y_no_steps
                self.posizione = self.save_position(x,y)
          # return the values for the global values
        print("Posizione iniziale: " + str(x) + " " + str(y))
        print("Fine posizionamento al centro\n")
        
    def coordinates(self, line):
        i=0
        #Leggo la coordinata x
        x=''
        negativo = False
        while (not (line[i]==",")):
            if(line[i] == '-'):
                negativo = True
            else:
                x = x + line[i]
            i += 1
        if negativo:
            x = int(x) * -1
        else:
            x = int(x)
            
        negativo = False
        i= i + 1
        #Leggo la coordinata y
        y=''
        while (not (line[i]==",")):
            if(line[i] == '-'):
                negativo = True
            else:
                y = y + line[i]
            i += 1
        if negativo:
            y = int(y) * -1
        else:
            y = int(y)
            
        return x, y
    
    def fix_position(self): #Aggiusto la posizione all'inizio
        print("Inizio posizionamento al centro\n")
        file = open("/home/pi/Desktop/Playbot4All-main/Motors_code/valore_posizione.txt", "r")
        posizione = file.readline()
        try: #Nel caso in valore posizione non sia stato salvato alcun valore, salva 1 di default
            self.posizione = int(posizione[0])
        except:
            self.posizione = 1;
        file.close()
        file = open("/home/pi/Desktop/Playbot4All-main/Motors_code/positions" + str(self.posizione) + ".txt", "r")            
        line = file.readline()
        file.close()
        try:
            #Leggo la posizione x e y della struttura
            x, y = self.coordinates(line)
            
            #Mi sposto al centro
            x_distance = 0 - x
            y_distance = 0 - y
            self.move_and_save(x_distance, y_distance, x, y)
        
        except: #errore nel salvataggio della posizione, riprovo con l'altro file
            if (self.posizione == 1):     
                file = open("/home/pi/Desktop/Playbot4All-main/Motors_code/positions2.txt", "r")            
            else:
                file = open("/home/pi/Desktop/Playbot4All-main/Motors_code/positions1.txt", "r")
            line = file.readline()
            file.close()
            
            x, y  = self.coordinates(line)
            
            x_distance = 0 - x
            y_distance = 0 - y
            self.move_and_save(x_distance, y_distance, x, y)
