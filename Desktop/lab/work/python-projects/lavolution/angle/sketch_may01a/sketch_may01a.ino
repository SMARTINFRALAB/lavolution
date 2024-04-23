#include<bluefruit.h>
#include <Adafruit_LSM6DS33.h>

float mv_per_lsb = 3600.0F/16384.0F/1000;
uint8_t beaconUuid[16] = 
{ 
  0xE0, 0xC5, 0x6D, 0xB5, 0xDF, 0xFB, 0x48, 0xD2, 
  0xB0, 0x60, 0xD0, 0xF5, 0xA7, 0x10, 0x96, 0xE0, 
};
BLEBeacon beacon(beaconUuid);

int adc_conc = A5;
float V_conc = 0, V_batt = 3.284; // 단위 : V 원래는 3.301
int Raw_conc = 0, Raw_batt = 0;
uint16_t Rconc = 0;

// 제일 중요한 R1 저항값 측정
// [k ohm] 사용하기 전 매번 멀티미터로 값을 측정한 뒤에 입력할것
// 저항 측정시에는 전원 연결을 해제하고 측정해야함
int R1 = 1001;

void setup() {
  Serial.begin(115200);
  analogReadResolution(14);
  Bluefruit.begin();
  Bluefruit.setTxPower(0);
  beacon.setManufacturer(0x004C);
  beacon.setRssiAt1m(-54);
  startAdv();
}

void loop()
{
  // 콘크리트에 걸리는 전압을 A5핀에서 읽음
  Raw_conc = analogRead(adc_conc);

  // 아날로그 출력 해상도를 2^14로 설정했으니 원래대로 스케일링
  V_conc = (float)Raw_conc * mv_per_lsb;  

  // 콘크리트, 배터리 전압 출력; 단위:[V]
  Serial.print("Concrete voltage and Battery voltage are: ");
  Serial.print(V_conc);
  Serial.print(", ");
  Serial.println(V_batt);
 
  // 콘크리트 저항 계산; 단위:[K Ohm]
  Rconc = (V_conc*R1)/(V_batt-V_conc); //kohm unit
  Serial.print("Concrete Resistance : ");
  Serial.println(Rconc); Serial.println(" ");

  // 비콘 전송 신호
  // 콘크리트 저항단위 : K Ohm, 콘크리트 전압단위 : V
  beacon.setMajorMinor(__swap16(Rconc),__swap16((int)(V_conc*1000)));
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
