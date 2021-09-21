#include <Servo.h>

Servo myServo;

int milli2secConv = 1000;

int test = 0;

void setup() {
  Serial.begin(9600);
  myServo.attach(18);
  myServo.writeMicroseconds(1000);
  
}

void loop() {
  // put your main code here, to run repeatedly:

  // simple code where I manually changed the values to see if I can dial into the right period
  myServo.writeMicroseconds(1200);
  delay(2*milli2secConv);
  myServo.writeMicroseconds(1500);
  delay(3*milli2secConv);
//  myServo.writeMicroseconds(1000);
//  delay(2*milli2secConv);
  myServo.writeMicroseconds(1800);
  delay(3*milli2secConv);
  
  myServo.write(0);
  delay(2*milli2secConv);

  // test to iterate through the different speeds to see where the motor changed behavior  
  for(test = 0; test < 180; test++){
    myServo.write(test);
    Serial.println(test);
    delay(1*milli2secConv);
  }
}
