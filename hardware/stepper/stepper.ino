#include <AccelStepper.h>

#define STEP_PIN 4
#define DIR_PIN 7
#define EN_PIN 8

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

const int stepsPerMove = 20; 

void setup() {
  pinMode(EN_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW);

  stepper.setMaxSpeed(100.0);
  
  stepper.setAcceleration(100.0);

  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    if (cmd == 'R' || cmd == 'r') {
      stepper.move(stepsPerMove);
      stepper.runToPosition();
      
      delay(750);
      
      Serial.println("D");
    }
  }
}