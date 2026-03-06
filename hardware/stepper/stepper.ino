#include <AccelStepper.h>

#define STEP_PIN 2
#define DIR_PIN 5

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

const int stepsPerMove = 10; 

void setup() {
  stepper.setMaxSpeed(500.0);
  stepper.setAcceleration(250.0);

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