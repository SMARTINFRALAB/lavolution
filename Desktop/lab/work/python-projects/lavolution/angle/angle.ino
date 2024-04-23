#include <Adafruit_LSM6DS33.h>    // Library for support of LSM6DS33 chip
Adafruit_LSM6DS33 accelerometer;  // Create an accelerometer object
char cmd;

void setup() 
{
  Serial.begin(115200);
  accelerometer.begin_I2C();        //  Start the I2C interface to the sensors
}


void loop() 
{
  float ax, ay, az;
  sensors_event_t accel, gyro, temp;

  accelerometer.getEvent(&accel, &gyro, &temp);  //  get the data

  ax = accel.acceleration.x;   //  Copy to convenient variables. Not necessary
  ay = accel.acceleration.y;
  az = accel.acceleration.z;

  if(Serial.available()){
    cmd = Serial.read();
    if(cmd=='a'){
      Serial.print(ax);
      Serial.print(',');
      Serial.print(ay);
      Serial.print(',');
      Serial.println(az);
      delay(100);
    }
  }
}
