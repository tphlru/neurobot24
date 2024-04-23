#include <ServoSmooth.h>
ServoSmooth servos[2];

bool ready_x, ready_y;

uint32_t servoTimer;
uint32_t turnTimer;

int deg_x = 90;
int deg_y = 90;

void setup() {
  Serial.begin(9600);
  pinMode(8, OUTPUT);
  
  servos[0].attach(9, 90);
  servos[0].smoothStart();
  servos[1].attach(10, 90);
  servos[1].smoothStart();

  servos[0].setSpeed(80);
  servos[1].setSpeed(80);
}

void loop() {
  String got;
  if (Serial.available()){
    got = Serial.readStringUntil('e');
    if (got.indexOf('s') >= 0){
      Serial.println("OK");
    }
    digitalWrite(8, LOW);
    int xi = got.indexOf('x');
    int yi = got.indexOf('y');
    deg_x = (got.substring(xi+1, yi)).toInt();
    deg_x = 180 - map(deg_x, -90, 90, 0, 180);
    deg_y = (got.substring(yi+1)).toInt();
    deg_y = 180 - map(deg_y, -90, 90, 0, 180);
    
    servos[0].setTargetDeg(deg_x);
    servos[1].setTargetDeg(deg_y);    
  }
  
  if (millis() - servoTimer >= 20) {
    servoTimer += 20;
    ready_x = servos[0].tickManual();
    ready_y = servos[1].tickManual();
  }

  if (ready_x == true and ready_y == true){
    digitalWrite(8, HIGH);
  }
  else{
    digitalWrite(8, LOW);
  }
}
