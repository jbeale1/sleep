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

const int CS_PIN  = 19;
const int DRDY_PIN = 18;
const int NUM_AVG = 5;

inline void waitDRDY() { while (digitalRead(DRDY_PIN) == HIGH) {} }

// Write MUX register without any delays (CS must already be LOW)
inline void fastSetMUX(uint8_t mux) {
    SPI.transfer(0x51); // WREG MUX_REG (0x50 | 0x01)
    SPI.transfer(0x00);
    SPI.transfer(mux);
}

long readPipelined(uint8_t nextMux) {
    // On entry: a conversion is already running
    waitDRDY();
    fastSetMUX(nextMux);           // switch to next channel
    SPI.transfer(0b11111100);      // SYNC
    delayMicroseconds(4);
    SPI.transfer(0b11111111);      // WAKEUP  (starts conversion on nextMux)
    SPI.transfer(0b00000001);      // RDATA   (reads result from PREVIOUS channel)
    delayMicroseconds(7);
    uint8_t b0 = SPI.transfer(0);
    uint8_t b1 = SPI.transfer(0);
    uint8_t b2 = SPI.transfer(0);
    long val = ((long)b0 << 16) | ((long)b1 << 8) | b2;
    return (val & (1l << 23)) ? val - 0x1000000 : val;
}

float toVolts(long raw) {
    // PGA=1, Vref=2.5V: full scale = ±2.5V = ±8388608 counts
    return raw * (2.0f * 2.5f / 8388608.0f);
}

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);

    ads.InitializeADC();
    ads.setDRATE(DRATE_2000SPS);
    ads.setPGA(PGA_1);

    Serial.println("ch0_V,ch1_V");

    // Start the pipeline: set MUX to ch0, begin first conversion
    SPI.beginTransaction(SPISettings(1920000, MSBFIRST, SPI_MODE1));
    digitalWrite(CS_PIN, LOW);
    fastSetMUX(SING_0);
    SPI.transfer(0b11111100); // SYNC
    delayMicroseconds(4);
    SPI.transfer(0b11111111); // WAKEUP
    // CS stays LOW — we keep the pipeline running
}

void loop() {
    float sumCh0 = 0, sumCh1 = 0;

    for (int i = 0; i < NUM_AVG; i++) {
        // Read ch0 result, set up ch1 conversion
        long ch0 = readPipelined(SING_1);
        // Read ch1 result, set up ch0 conversion
        long ch1 = readPipelined(SING_0);
        sumCh0 += toVolts(ch0);
        sumCh1 += toVolts(ch1);
    }

    Serial.print(sumCh0 / NUM_AVG, 6);
    Serial.print(",");
    Serial.println(sumCh1 / NUM_AVG, 6);
}
