// Arduino code for Sparkfun Thing Plus RP2040 to read from Adafruit DPS310 
// connected via the onboard JST SH 4-pin I2C connector (Qwiic / Stemma QT) 
// print via USB serial and log to SD card. Uses Wire1 for I2C and SPI1 for SD card.
// 2026-02-08 J.Beale 

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_DPS310.h>
#include <SPI.h>
#include <SD.h>

static const int SD_CS = 9;
static const int LED_PIN = 25;

Adafruit_DPS310 dps;
File logFile;
char filename[16];

static const int BUF_SIZE = 320; // 10 seconds of data at 32 Hz 
float pBuf[BUF_SIZE];
uint32_t tBuf[BUF_SIZE];
int bufIdx = 0;

unsigned long lastBlink = 0;
static const unsigned long BLINK_INTERVAL = 1000;

void updateLED() {
  unsigned long now = millis();
  if (now - lastBlink >= BLINK_INTERVAL) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    lastBlink = now;
  }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);

  unsigned long t0 = millis();
  while (!Serial && (millis() - t0 < 3000)) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(100);
  }

  digitalWrite(LED_PIN, HIGH);
  Serial.println("1: Serial OK");

  Wire1.setSDA(6);
  Wire1.setSCL(7);
  Wire1.begin();
  Serial.println("2: Wire OK");

  if (!dps.begin_I2C(0x77, &Wire1)) {
    Serial.println("DPS310 not found! Trying 0x76...");
    if (!dps.begin_I2C(0x76, &Wire1)) {
      Serial.println("DPS310 not found at either address!");
      while (1) {
          digitalWrite(LED_PIN, !digitalRead(LED_PIN));
          delay(100);
      }
    }
  }
  Serial.println("3: DPS310 OK");

  dps.configurePressure(DPS310_32HZ, DPS310_8SAMPLES);
  dps.configureTemperature(DPS310_1HZ, DPS310_1SAMPLE);
  dps.setMode(DPS310_CONT_PRESSURE);
  
  delay(500);  // the first few readings are bad
  // Discard the first few readings as they are often inaccurate
  sensors_event_t t, p;
  for (int i = 0; i < 5; i++) {
      while (!dps.pressureAvailable()) delay(1);
      dps.getEvents(&t, &p);
  }

  Serial.println("4: DPS310 configured");

  SPI1.setRX(12);
  SPI1.setTX(15);
  SPI1.setSCK(14);
  if (!SD.begin(SD_CS, SPI1)) {
    Serial.println("SD init failed!");
    while (1) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(100);
  }
  }
  Serial.println("5: SD OK");

  for (int i = 0; i < 1000; i++) {
    snprintf(filename, sizeof(filename), "LOG_%03d.csv", i);
    if (!SD.exists(filename)) break;
  }

  logFile = SD.open(filename, FILE_WRITE);
  if (!logFile) {
    Serial.println("Failed to open log file!");
    while (1) {
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));
        delay(100);
    }
  }

  logFile.println("millis,pressure_hPa");
  logFile.flush();

  Serial.print("Logging to: ");
  Serial.println(filename);
}

void flushBuffer() {
  if (bufIdx == 0) return;
  for (int i = 0; i < bufIdx; i++) {
    logFile.print(tBuf[i]);
    logFile.print(',');
    logFile.println(pBuf[i], 4);
  }
  logFile.flush();
  bufIdx = 0;
}

void loop() {
  updateLED();
  sensors_event_t temp_event, pressure_event;

  if (dps.pressureAvailable()) {
    dps.getEvents(&temp_event, &pressure_event);

    uint32_t now = millis();
    float p = pressure_event.pressure;

    tBuf[bufIdx] = now;
    pBuf[bufIdx] = p;
    bufIdx++;

    Serial.println(p, 4);

    if (bufIdx >= BUF_SIZE) {
      flushBuffer();
    }
  }
}
