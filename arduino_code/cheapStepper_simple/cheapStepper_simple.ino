#include <CheapStepper.h>
#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"

MPU6050 mpu;

#define degK 11.380

CheapStepper stepper;


uint8_t fifoBuffer[45];         // буфер
int stepsleft = 0;
bool moveClockwise;
float set = 20.5;

float y;



float get_y() {
  float ypr[3];
  if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
    Quaternion q;
    VectorFloat gravity;

    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);

  }
  return constrain(ypr[0] * float(57.296) + 90, -180, 180);
}


void setup() {
  Serial.begin(9600);
  stepper.setRpm(12);
  Wire.begin();
  Wire.setClock(1000000UL);   // разгоняем шину на максимум

  // инициализация DMP
  mpu.initialize();
  mpu.dmpInitialize();
  mpu.setDMPEnabled(true);
  y = get_y();
}

void loop() {
  if (y > set && abs(set-y) > 0.15 && millis() > 10000) {
    Serial.println(y);
    stepper.step(true);
    y = get_y();
    delay(50);
  }
  else if (y < set && abs(set-y) > 0.15 && millis() > 10000) {
    Serial.println(y);
    stepper.step(false);
    y = get_y();
    delay(50);
  }
  else {
    y = get_y();
    delay(100);
  }
}
