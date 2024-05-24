#include <ServoSmooth.h>
ServoSmooth servos[2];

bool ready_x, ready_y;

uint32_t servoTimer;
uint32_t turnTimer;

const float bx = 90;
const float by = 90 + 8;


float deg_x = bx;
float deg_y = by;

void setup() {
  Serial.begin(9600);
  pinMode(8, OUTPUT);
  pinMode(13, OUTPUT);
  pinMode(4, INPUT_PULLUP);

  servos[0].attach(9, bx);
  servos[0].smoothStart();
  servos[1].attach(10, by);
  servos[1].smoothStart();

  servos[0].setSpeed(80);
  servos[1].setSpeed(80);

  servos[0].setTargetDeg(bx);
  servos[1].setTargetDeg(by);
}

void loop() {
  String got;
  if (digitalRead(4) == LOW) {
    Serial.println("btn");
    delay(1500);
  }
  if (Serial.available()) {
    got = Serial.readStringUntil('e');
    if (got.indexOf('s') >= 0) {
      Serial.println("OK");
    }
    digitalWrite(8, LOW);
    int xi = got.indexOf('x');
    int yi = got.indexOf('y');
    deg_x = (got.substring(xi + 1, yi)).toInt();
    deg_x = bx + deg_x;
    deg_y = (got.substring(yi + 1)).toInt();
    deg_y = by + deg_y;

    deg_x = 180 - constrain(deg_x, 0, 180);
    deg_y = constrain(deg_y, 0, 180);

    servos[0].setTargetDeg(deg_x);
    servos[1].setTargetDeg(deg_y);
  }

  if (millis() - servoTimer >= 30) {
    servoTimer += 30;
    ready_x = servos[0].tickManual();
    ready_y = servos[1].tickManual();
  }

  if (ready_x == true and ready_y == true) {
    digitalWrite(8, HIGH);
    digitalWrite(13, HIGH);
  } else {
    digitalWrite(8, LOW);
    digitalWrite(13, LOW);
  }
}