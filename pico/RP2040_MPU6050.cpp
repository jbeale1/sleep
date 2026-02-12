// Arduino code for Sparkfun Thing Plus RP2040 to read from MPU-6050 
// accelerometer + gyroscope connected via onboard Qwiic I2C connector.
// Calibrates at startup, measures noise floor, then continuously reports
// pitch, roll, integrated rotation, total linear impulse, and accel RMS.
// 5 Hz output rate for breathing + cardiac band capture.
// Prints via USB serial and logs to SD card.
// Uses Wire1 for I2C and SPI1 for SD card.
// 2026-02-11 J.Beale

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
    greiman/SdFat
monitor_speed = 115200

*/


#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <SPI.h>
#include <SdFat.h>
#include <math.h>

static const int SD_CS   = 9;
static const int LED_PIN = 25;

static const int    CAL_N         = 2500;   // samples for offset calibration
static const int    NOISE_WINDOWS = 40;     // windows to characterize noise floor
static const float  AVG_TIME      = 0.2f;   // seconds per measurement window (5 Hz output)
static const float  GRAVITY       = 9.806f;
static const float  NF_SCALE      = 0.5f;   // noise floor subtraction fraction (0=none, 1=full)

Adafruit_MPU6050 mpu;
SdFat sd;
FsFile logFile;
char filename[40];  // long filename: MOT_2026-02-09_143022.csv

// calibration offsets
float cx, cy, cz;
float cgx, cgy, cgz;

// millis() value when PC timestamp was received (0 if no timestamp)
unsigned long millis_at_time_sync = 0;

// Crystal drift correction: positive = crystal runs slow (RP2040 lags wall clock).
// Measured via drift log against NTP-synced PC. Adjust to match your board/temperature.
static const float PPM_CORRECTION = 10.6f;

// Returns drift-corrected elapsed ms since time sync.
// Compensates for crystal frequency error so reported time tracks wall clock.
unsigned long correctedElapsed() {
  unsigned long raw = millis() - millis_at_time_sync;
  return raw + (unsigned long)(raw * (PPM_CORRECTION / 1e6f));
}

// noise floor values
float nf_rot, nf_total, nf_rms;

// --- LED timecode (runs on core 1) ---
// Error blinks (core 0, setup only - before core 1 takes over):
// 1 blink  = MPU-6050 not found
// 2 blinks = SD init failed
// 3 blinks = log file open failed
// Timecode (core 1, after setup):
// :00 even min = long-long    :00 odd min = long-short
// :15 = 1 short   :30 = 2 short   :45 = 3 short
// 40ms tick at start of each second, except dark zones around marks.
// Dark zone before :15,:30,:45 = 2s. Before :00 = 2+minute_pair (2-6s),
// encoding position within 10-minute cycle.  1s dark after all marks.
// No sync = 1Hz heartbeat.

static const int BLINK_SHORT = 80;   // ms
static const int BLINK_LONG  = 300;  // ms
static const int BLINK_TICK  = 40;   // ms — second tick
static const int BLINK_GAP   = 150;  // ms between blinks in a group

// shared state: written by core 0 in setup, read by core 1
volatile bool     timecode_active = false;  // core 1 takes over LED
volatile bool     has_time_sync   = false;
volatile long     sync_second_of_day = 0;   // HH*3600+MM*60+SS at sync time

// blink N times, pause, repeat forever (core 0, setup only)
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

// --- Core 1: LED timecode ---
void blinkShort() {
  digitalWrite(LED_PIN, HIGH);
  delay(BLINK_SHORT);
  digitalWrite(LED_PIN, LOW);
}

void blinkLong() {
  digitalWrite(LED_PIN, HIGH);
  delay(BLINK_LONG);
  digitalWrite(LED_PIN, LOW);
}

void blinkTick() {
  digitalWrite(LED_PIN, HIGH);
  delay(BLINK_TICK);
  digitalWrite(LED_PIN, LOW);
}

// returns true if sec_in_min is in the exclusion zone around a timecode mark
// (but not on the mark itself). Marks 15,30,45: 2 before, 1 after.
// Mark 0: variable dark before (2 + minute_pair), 1 after.
bool nearMark(int s, int minute) {
  // marks 15, 30, 45: fixed 2 before, 1 after
  for (int mark : {15, 30, 45}) {
    int diff = s - mark;
    if (diff >= -2 && diff <= 1 && diff != 0) return true;
  }
  // mark 0: variable dark before, 1 after
  int dark_before = 2 + (minute % 10) / 2;
  int diff = s;  // relative to :00
  if (diff > 30) diff -= 60;  // wrap: 59 -> -1, 58 -> -2, etc.
  if (diff >= -dark_before && diff <= 1 && diff != 0) return true;
  return false;
}

void setup1() {
  // nothing needed — LED pin configured by core 0
}

void loop1() {
  if (!timecode_active) {
    delay(100);
    return;
  }

  if (!has_time_sync) {
    // no timestamp — simple 1Hz heartbeat
    digitalWrite(LED_PIN, HIGH);
    delay(500);
    digitalWrite(LED_PIN, LOW);
    delay(500);
    return;
  }

  // compute current wall-clock position (drift-corrected)
  unsigned long elapsed_ms = correctedElapsed();
  long current_sod = sync_second_of_day + (long)(elapsed_ms / 1000);
  // handle day wrap (>86400) just in case
  current_sod = current_sod % 86400;
  int sec_in_min = (int)(current_sod % 60);
  int minute     = (int)((current_sod / 60) % 60);

  static long last_blink_sod = -1;

  // only fire each pattern once per second
  if (current_sod == last_blink_sod) {
    delay(10);
    return;
  }

  if (sec_in_min == 0) {
    last_blink_sod = current_sod;
    if (minute % 2 == 0) {
      blinkLong(); delay(BLINK_GAP); blinkLong();       // even: long-long
    } else {
      blinkLong(); delay(BLINK_GAP); blinkShort();      // odd: long-short
    }
  } else if (sec_in_min == 15) {
    last_blink_sod = current_sod;
    blinkShort();                                        // 1 short
  } else if (sec_in_min == 30) {
    last_blink_sod = current_sod;
    blinkShort(); delay(BLINK_GAP); blinkShort();        // 2 short
  } else if (sec_in_min == 45) {
    last_blink_sod = current_sod;
    blinkShort(); delay(BLINK_GAP); blinkShort();
    delay(BLINK_GAP); blinkShort();                      // 3 short
  } else {
    last_blink_sod = current_sod;
    // tick blink on seconds outside exclusion zone around timecode marks
    if (!nearMark(sec_in_min, minute)) {
      blinkTick();
    }
    // wait for next second boundary (using corrected time)
    unsigned long now_ms = correctedElapsed();
    unsigned long ms_into_sec = now_ms % 1000;
    if (ms_into_sec < 990) {
      delay(990 - ms_into_sec);  // sleep until just before next second
    }
    // spin-wait the last few ms for tight alignment
    unsigned long next_sec_ms = (now_ms / 1000 + 1) * 1000;
    while (correctedElapsed() < next_sec_ms) { /* spin */ }
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
  Serial.println("=== RP2040_MPU6050 v3.1 ===");
  Serial.print("Boot at ");
  Serial.print(millis());
  Serial.println(" ms");
  Serial.print("Window: ");
  Serial.print(AVG_TIME * 1000, 0);
  Serial.print(" ms  Output rate: ");
  Serial.print(1.0f / AVG_TIME, 1);
  Serial.print(" Hz  PPM correction: ");
  Serial.println(PPM_CORRECTION, 1);

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

  // --- SD card init (SdFat on SPI1) ---
  Serial.println("4: Init SD (CS=9, SPI1)...");
  SPI1.setRX(12);
  SPI1.setTX(15);
  SPI1.setSCK(14);
  SdSpiConfig sdConfig(SD_CS, DEDICATED_SPI, SD_SCK_MHZ(25), &SPI1);
  if (!sd.begin(sdConfig)) {
    Serial.println("   SD FAILED! (2 blinks)");
    errorBlink(2);
  }
  Serial.println("   SD OK");

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
        millis_at_time_sync = millis();
        break;
      }
    }
    delay(1);
  }

  // --- Create log file ---
  // With timestamp: MOT_2026-02-09_143022.csv
  // Without:        MOT_000.csv (sequential)
  Serial.println("5: Creating log file...");
  if (got_time) {
    // normalize ISO 8601 'T' separator to space for sscanf
    for (char *p = timebuf; *p; p++) { if (*p == 'T') *p = ' '; }
    int yr, mo, dy, hh, mm, ss;
    if (sscanf(timebuf, "%d-%d-%d %d:%d:%d", &yr, &mo, &dy, &hh, &mm, &ss) >= 6) {
      snprintf(filename, sizeof(filename), "MOT_%04d-%02d-%02d_%02d%02d%02d.csv",
               yr, mo, dy, hh, mm, ss);
      sync_second_of_day = hh * 3600L + mm * 60L + ss;
      has_time_sync = true;
    } else {
      got_time = false;  // format not recognized, fall back
    }
  }
  if (!got_time) {
    for (int i = 0; i < 1000; i++) {
      snprintf(filename, sizeof(filename), "MOT_%03d.csv", i);
      if (!sd.exists(filename)) break;
    }
  }
  logFile = sd.open(filename, FILE_WRITE);
  if (!logFile) {
    Serial.print("   File open FAILED! (3 blinks): ");
    Serial.println(filename);
    errorBlink(3);
  }
  Serial.print("   Logging to: ");
  Serial.println(filename);

  if (got_time) {
    Serial.print("   Timestamp: ");
    Serial.print(timebuf);
    Serial.print(" (sync at millis=");
    Serial.print(millis_at_time_sync);
    Serial.println(")");
    logFile.print("# start ");
    logFile.print(timebuf);
    logFile.print(" sync_millis=");
    logFile.println(millis_at_time_sync);
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

  Serial.println("8: Running - LED timecode on core 1");
  timecode_active = true;
}

void loop() {
  WindowResult wr;
  measureWindow(wr);

  float rot   = max(0.0f, wr.rot_deg       - nf_rot   * NF_SCALE);
  float total = max(0.0f, wr.total_impulse  - nf_total * NF_SCALE);
  float rms   = max(0.0f, wr.accel_rms      - nf_rms   * NF_SCALE);

  uint32_t msec = correctedElapsed();

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
