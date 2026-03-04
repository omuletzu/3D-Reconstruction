#include <AccelStepper.h>

#define MotorInterfaceType 4

const int stepsPerMove = 114;

AccelStepper stepper(MotorInterfaceType, 8, 10, 9, 11);

void setup() {
  stepper.setMaxSpeed(1000.0);
  stepper.setAcceleration(500.0);

  Serial.begin(9600);
}

void loop() {
  if(Serial.available() > 0) {
    char cmd = Serial.read();

    if(cmd == 'R' || cmd == 'r') {
      stepper.step(stepsPerMove);

      delay(750);

      Serial.println("D");
    }
  }
}