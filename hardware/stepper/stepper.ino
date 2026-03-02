#include <Stepper.h>

const int totalRevolutionSteps = 2048;

const int stepsPerMove = 114;

Stepper stepper(stepsPerRevolution, 8, 10, 9, 11);

void setup() {
  stepper.setSpeed(15);

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