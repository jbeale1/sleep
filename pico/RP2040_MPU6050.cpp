// Arduino code for Sparkfun Thing Plus RP2040 to read from MPU-6050 
// accelerometer + gyroscope connected via onboard Qwiic I2C connector.
// Calibrates at startup, measures noise floor, then continuously reports
// pitch, roll, integrated rotation, total linear impulse, and accel RMS.
// 5 Hz output rate for breathing + cardiac band capture.
// Prints via USB serial and logs to SD card.
// Uses Wire1 for I2C and SPI1 for SD card.
// 2026-02-09 J.Beale

/*
use this platformio.ini for building and uploading:

[env:thingplus]
platform = https://github.com/maxgerhardt/platform-raspberrypi.git
board = sparkfun_thingplusrp2040
framework = arduino
board_build.core = earlephilhower
board_build.filesystem_size = 0
lib_deps =
    adafruit/Adafruit MPU6050
monitor_speed = 115200

*/


#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <SPI.h>
#include <SD.h>
#include <math.h>

static const int SD_CS   = 9;
static const int LED_PIN = 25;

static const int    CAL_N         = 2500;   // samples for offset calibration
static const int    NOISE_WINDOWS = 40;     // windows to characterize noise floor
static const float  AVG_TIME      = 0.2f;   // seconds per measurement window (5 Hz output)
static const float  GRAVITY       = 9.806f;
static const float  NF_SCALE      = 0.5f;   // noise floor subtraction fraction (0=none, 1=full)

Adafruit_MPU6050 mpu;
File logFile;
char filename[16];

// calibration offsets
float cx, cy, cz;
float cgx, cgy, cgz;

// noise floor values
float nf_rot, nf_total, nf_rms;

// --- LED blink patterns ---
// 1 blink  = MPU-6050 not found
// 2 blinks = SD init failed
// 3 blinks = log file open failed
// steady 1Hz toggle = running normally
unsigned long lastBlink = 0;
static const unsigned long BLINK_INTERVAL = 1000;

void updateLED() {
  unsigned long now = millis();
  if (now - lastBlink >= BLINK_INTERVAL) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    lastBlink = now;
  }
}

// blink N times, pause, repeat forever
void errorBlink(int count) {
  while (1) {
    for (int i = 0; i < count; i++) {
      digitalWrite(LED_PIN, HIGH);
      delay(150);
      digitalWrite(LED_PIN, LOW);
      delay(150);
    }
    delay(1000);
  }
}

// --- Per-sample storage (allocated once) ---
struct Sample {
  float ax, ay, az;
  float gx, gy, gz;
  float t;  // timestamp in seconds
};

static const int MAX_SAMPLES = 256;  // plenty for 0.2s window
Sample samples[MAX_SAMPLES];

// --- Measurement window result ---
struct WindowResult {
  float pitch, roll;
  float rot_deg;       // integrated rotation in degrees
  float total_impulse; // sum of all |accel residual| * dt (mG·s)
  float accel_rms;     // RMS of accel residual magnitude (mG)
  int n;
};

void measureWindow(WindowResult &r) {
  int n = 0;
  float deadline = millis() / 1000.0f + AVG_TIME;
  sensors_event_t a, g, temp;

  while ((millis() / 1000.0f) < deadline && n < MAX_SAMPLES) {
    mpu.getEvent(&a, &g, &temp);
    float now = millis() / 1000.0f;
    samples[n].ax = a.acceleration.x - cx;
    samples[n].ay = a.acceleration.y - cy;
    samples[n].az = a.acceleration.z - cz;
    samples[n].gx = g.gyro.x - cgx;
    samples[n].gy = g.gyro.y - cgy;
    samples[n].gz = g.gyro.z - cgz;
    samples[n].t  = now;
    n++;
  }

  // compute average accel
  float avg_x = 0, avg_y = 0, avg_z = 0;
  for (int i = 0; i < n; i++) {
    avg_x += samples[i].ax;
    avg_y += samples[i].ay;
    avg_z += samples[i].az;
  }
  avg_x /= n;
  avg_y /= n;
  avg_z /= n;

  float total_pos = 0;  // total impulse (positive accumulation of all axes)
  float integ_angle = 0;
  float sum_sq = 0;     // for RMS of accel residual magnitude

  for (int i = 1; i < n; i++) {
    float dt = samples[i].t - samples[i - 1].t;

    float lx = samples[i].ax - avg_x;
    float ly = samples[i].ay - avg_y;
    float lz = samples[i].az - avg_z;

    // total impulse: sum of absolute deviations * dt
    total_pos += (fabsf(lx) + fabsf(ly) + fabsf(lz)) * dt;

    // accel residual magnitude squared (for RMS, in m/s^2)
    sum_sq += lx * lx + ly * ly + lz * lz;

    float rx = samples[i].gx;
    float ry = samples[i].gy;
    float rz = samples[i].gz;
    float omega = sqrtf(rx * rx + ry * ry + rz * rz);
    integ_angle += omega * dt;
  }

  float denom_p = copysignf(sqrtf(avg_y * avg_y + avg_z * avg_z), avg_z);
  float denom_r = copysignf(sqrtf(avg_x * avg_x + avg_z * avg_z), avg_z);
  r.pitch = atan2f(avg_x, denom_p) * 180.0f / (float)M_PI;
  r.roll  = atan2f(avg_y, denom_r) * 180.0f / (float)M_PI;
  r.rot_deg = integ_angle * 180.0f / (float)M_PI;
  r.total_impulse = total_pos * 1000.0f;  // convert to mG·s
  r.accel_rms = sqrtf(sum_sq / (n - 1)) * 1000.0f / GRAVITY;  // convert to mG
  r.n = n;
}

// --- SD card output buffer ---
static const int LOG_BUF_SIZE = 50;  // flush every 50 lines (10 seconds at 5 Hz)
char lineBuf[LOG_BUF_SIZE][80];
int logBufIdx = 0;

void flushLogBuffer() {
  if (logBufIdx == 0) return;
  for (int i = 0; i < logBufIdx; i++) {
    logFile.print(lineBuf[i]);
  }
  logFile.flush();
  logBufIdx = 0;
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
  Serial.begin(115200);

  unsigned long t0 = millis();
  while (!Serial && (millis() - t0 < 5000)) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(200);
  }
  digitalWrite(LED_PIN, HIGH);

  Serial.println();
  Serial.println("=== RP2040_MPU6050 v2.0 ===");
  Serial.print("Boot at ");
  Serial.print(millis());
  Serial.println(" ms");
  Serial.print("Window: ");
  Serial.print(AVG_TIME * 1000, 0);
  Serial.print(" ms  Output rate: ");
  Serial.print(1.0f / AVG_TIME, 1);
  Serial.println(" Hz");

  // --- I2C on Wire1 (Qwiic) ---
  Serial.println("1: Init Wire1 (SDA=6 SCL=7)...");
  Wire1.setSDA(6);
  Wire1.setSCL(7);
  Wire1.begin();
  Wire1.setClock(400000);
  Serial.println("   Wire1 OK");

  // --- I2C bus scan ---
  Serial.println("2: I2C scan on Wire1...");
  int nDevices = 0;
  for (byte addr = 1; addr < 127; addr++) {
    Wire1.beginTransmission(addr);
    byte error = Wire1.endTransmission();
    if (error == 0) {
      Serial.print("   Found device at 0x");
      Serial.println(addr, HEX);
      nDevices++;
    }
  }
  Serial.print("   ");
  Serial.print(nDevices);
  Serial.println(" device(s) found");

  // --- MPU-6050 init ---
  Serial.println("3: Init MPU-6050...");
  if (!mpu.begin(0x68, &Wire1)) {
    Serial.println("   MPU-6050 FAILED! (1 blink)");
    errorBlink(1);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("   MPU-6050 OK");

  // --- SD card init ---
  Serial.println("4: Init SD (CS=9, SPI1)...");
  SPI1.setRX(12);
  SPI1.setTX(15);
  SPI1.setSCK(14);
  if (!SD.begin(SD_CS, SPI1)) {
    Serial.println("   SD FAILED! (2 blinks)");
    errorBlink(2);
  }
  Serial.println("   SD OK");

  // --- Create log file ---
  Serial.println("5: Creating log file...");
  for (int i = 0; i < 1000; i++) {
    snprintf(filename, sizeof(filename), "MOT_%03d.csv", i);
    if (!SD.exists(filename)) break;
  }
  logFile = SD.open(filename, FILE_WRITE);
  if (!logFile) {
    Serial.print("   File open FAILED! (3 blinks): ");
    Serial.println(filename);
    errorBlink(3);
  }
  Serial.print("   Logging to: ");
  Serial.println(filename);

  // --- Request timestamp from PC ---
  Serial.println("TIME?");
  char timebuf[40] = {0};
  bool got_time = false;
  unsigned long t_req = millis();
  while (millis() - t_req < 3000) {
    if (Serial.available()) {
      String line = Serial.readStringUntil('\n');
      line.trim();
      if (line.startsWith("TIME=") && line.length() > 5) {
        line.substring(5).toCharArray(timebuf, sizeof(timebuf));
        got_time = true;
        break;
      }
    }
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(50);
  }
  if (got_time) {
    Serial.print("   Timestamp: ");
    Serial.println(timebuf);
    logFile.print("# start ");
    logFile.println(timebuf);
  } else {
    Serial.println("   No timestamp received (standalone mode)");
    logFile.print("# start unknown, boot_millis=");
    logFile.println(millis());
  }
  logFile.flush();

  // --- Phase 1: Calibrate sensor offsets ---
  Serial.print("6: Calibrating offsets (");
  Serial.print(CAL_N);
  Serial.println(" samples)...");
  cx = cy = cz = 0;
  cgx = cgy = cgz = 0;
  sensors_event_t a, g, temp;
  for (int i = 0; i < CAL_N; i++) {
    mpu.getEvent(&a, &g, &temp);
    cx  += a.acceleration.x;
    cy  += a.acceleration.y;
    cz  += a.acceleration.z;
    cgx += g.gyro.x;
    cgy += g.gyro.y;
    cgz += g.gyro.z;
    if (i % 500 == 0) {
      updateLED();
      Serial.print("   ");
      Serial.print(i);
      Serial.print("/");
      Serial.println(CAL_N);
    }
  }
  cx  /= CAL_N;
  cy  /= CAL_N;
  cz   = cz / CAL_N - GRAVITY;
  cgx /= CAL_N;
  cgy /= CAL_N;
  cgz /= CAL_N;
  Serial.print("   Accel offsets: ");
  Serial.print(cx, 3); Serial.print(", ");
  Serial.print(cy, 3); Serial.print(", ");
  Serial.println(cz, 3);
  Serial.print("   Gyro offsets:  ");
  Serial.print(cgx, 4); Serial.print(", ");
  Serial.print(cgy, 4); Serial.print(", ");
  Serial.println(cgz, 4);

  // --- Phase 2: Characterize noise floor ---
  Serial.print("7: Measuring noise floor (");
  Serial.print(NOISE_WINDOWS);
  Serial.println(" windows)...");
  float sum_rot = 0, sum_total = 0, sum_rms = 0;
  WindowResult wr;
  for (int w = 0; w < NOISE_WINDOWS; w++) {
    measureWindow(wr);
    sum_rot   += wr.rot_deg;
    sum_total += wr.total_impulse;
    sum_rms   += wr.accel_rms;
    updateLED();
    if (w % 10 == 0) {
      Serial.print("   window ");
      Serial.print(w);
      Serial.print("/");
      Serial.println(NOISE_WINDOWS);
    }
  }
  nf_rot   = sum_rot   / NOISE_WINDOWS;
  nf_total = sum_total / NOISE_WINDOWS;
  nf_rms   = sum_rms   / NOISE_WINDOWS;
  Serial.print("   NF rot=");   Serial.print(nf_rot, 4);
  Serial.print(" total="); Serial.print(nf_total, 2);
  Serial.print(" rms=");   Serial.println(nf_rms, 2);

  // --- Write CSV header ---
  const char *header = "msec,pitch,roll,rot,total,rms\n";
  logFile.print(header);
  logFile.flush();
  Serial.print(header);

  Serial.println("8: Running - steady 1Hz blink = OK");
}

void loop() {
  updateLED();

  WindowResult wr;
  measureWindow(wr);

  float rot   = max(0.0f, wr.rot_deg       - nf_rot   * NF_SCALE);
  float total = max(0.0f, wr.total_impulse  - nf_total * NF_SCALE);
  float rms   = max(0.0f, wr.accel_rms      - nf_rms   * NF_SCALE);

  uint32_t msec = millis();

  // format output line: 6 columns, compact
  char line[80];
  snprintf(line, sizeof(line),
    "%lu,%.3f,%.3f,%.2f,%.1f,%.1f\n",
    (unsigned long)msec,
    wr.pitch, wr.roll, rot,
    total, rms);

  Serial.print(line);

  // buffer to SD
  strncpy(lineBuf[logBufIdx], line, sizeof(lineBuf[0]));
  logBufIdx++;
  if (logBufIdx >= LOG_BUF_SIZE) {
    flushLogBuffer();
  }
}

