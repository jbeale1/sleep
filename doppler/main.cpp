#include <Arduino.h>
#include <ADS1256.h>
#include <SPI.h>

/*
PlatformIO Project Configuration File:

[env:waveshare_rp2040_zero]
platform = https://github.com/maxgerhardt/platform-raspberrypi.git
board = waveshare_rp2040_zero
framework = arduino
monitor_speed = 115200
lib_deps =
    https://github.com/CuriousScientist0/ADS1256.git

*/

ADS1256 ads(18, 20, 21, 19, 2.5);

const int CS_PIN   = 19;
const int DRDY_PIN = 18;
const int NUM_AVG  = 5;

uint32_t seq = 0;

inline void waitDRDY() { while (digitalRead(DRDY_PIN) == HIGH) {} }

inline void fastSetMUX(uint8_t mux) {
    SPI.transfer(0x51);
    SPI.transfer(0x00);
    SPI.transfer(mux);
}

long readPipelined(uint8_t nextMux) {
    waitDRDY();
    fastSetMUX(nextMux);
    SPI.transfer(0b11111100);      // SYNC
    delayMicroseconds(4);
    SPI.transfer(0b11111111);      // WAKEUP
    SPI.transfer(0b00000001);      // RDATA
    delayMicroseconds(7);
    uint8_t b0 = SPI.transfer(0);
    uint8_t b1 = SPI.transfer(0);
    uint8_t b2 = SPI.transfer(0);
    long val = ((long)b0 << 16) | ((long)b1 << 8) | b2;
    return (val & (1l << 23)) ? val - 0x1000000 : val;
}

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);

    ads.InitializeADC();
    ads.setDRATE(DRATE_2000SPS);
    ads.setPGA(PGA_1);

    Serial.println("seq,ms,I,Q");

    SPI.beginTransaction(SPISettings(1920000, MSBFIRST, SPI_MODE1));
    digitalWrite(CS_PIN, LOW);
    fastSetMUX(SING_0);
    SPI.transfer(0b11111100);      // SYNC
    delayMicroseconds(4);
    SPI.transfer(0b11111111);      // WAKEUP
}

void loop() {
    long sumCh0 = 0, sumCh1 = 0;

    for (int i = 0; i < NUM_AVG; i++) {
        sumCh0 += readPipelined(SING_1);
        sumCh1 += readPipelined(SING_0);
    }

    Serial.print(seq++);
    Serial.print(',');
    Serial.print(millis() % 1000);
    Serial.print(',');
    Serial.print(sumCh0 / NUM_AVG);
    Serial.print(',');
    Serial.println(sumCh1 / NUM_AVG);
}
