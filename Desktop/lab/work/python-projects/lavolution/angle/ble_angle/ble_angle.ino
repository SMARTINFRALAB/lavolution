#include <Adafruit_LSM6DS33.h>    // Library for support of LSM6DS33 chip
#include<bluefruit.h>
#include <math.h>

Adafruit_LSM6DS33 accelerometer;  // Create an accelerometer object
uint8_t beaconUuid[16] = 
{ 
  0xE0, 0xC5, 0x6D, 0xB5, 0xDF, 0xFB, 0x48, 0xD2, 
  0xB0, 0x60, 0xD0, 0xF5, 0xA7, 0x10, 0x96, 0xE0, 
};
BLEBeacon beacon(beaconUuid);

void setup() 
{
  Serial.begin(115200);
  analogReadResolution(14);
  Bluefruit.begin();
  Bluefruit.setTxPower(0);
  beacon.setManufacturer(0x004C);
  beacon.setRssiAt1m(-54);
  startAdv();
  accelerometer.begin_I2C();        //  Start the I2C interface to the sensors
}


void loop() 
{
  float ax, ay, az, ang;
  sensors_event_t accel, gyro, temp;

  accelerometer.getEvent(&accel, &gyro, &temp);  //  get the data

  ax = accel.acceleration.x;   //  Copy to convenient variables. Not necessary
  ay = accel.acceleration.y;
  az = accel.acceleration.z;

  ang = degrees(atan2(ax,sqrt(pow(ay, 2)+pow(az, 2))))*100;

  Serial.print("Angle : ");
  Serial.println(ang); Serial.println(" ");

  beacon.setMajorMinor((uint16_t)ang, (uint16_t)ang);
  Bluefruit.Advertising.setBeacon(beacon);
  Bluefruit.Advertising.start(1);
  delay(400);
}

void startAdv(void)
{
  Bluefruit.Advertising.setBeacon(beacon);
  Bluefruit.Advertising.setInterval(800, 800);
  Bluefruit.Advertising.setFastTimeout(3);
}