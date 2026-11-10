/*
throw_esp32s3.ino - REAL THROW on an ESP32-S3 Super Mini instead of the Raspberry Pi (port of throw_real.py v d)
=====================================================================================================
Same detection, PPO catch, checks, beeps, hold 10 s -> LAND, ALT_HOLD after, take-over and result files as
throw_real.py on the Pi (logic: throw_core.cpp, tested 9 Nov 2026 against SITL built from your firmware).

0. SETTINGS (change here only)
------------------------------------------------------------------------------------------------------
    | Setting   | Value   | Meaning                                                                   |
    |-----------|---------|---------------------------------------------------------------------------|
    | FC_RX_PIN | 4       | ESP32 GPIO 4  <- FC TX3 (SERIAL2)                                         |
    | FC_TX_PIN | 5       | ESP32 GPIO 5  -> FC RX3 (SERIAL2)                                         |
    | FC_BAUD   | 921600  | = SERIAL2_BAUD 921 on the FC (MAVLink2, SERIAL2_PROTOCOL 2)               |
    Throw settings (CH5, 14.9 V, 5 sats, 10 s hold, ...) are at the top of throw_core.cpp (= throw_real.py).

1. HOW TO RUN  (Arduino IDE 2, board package "esp32 by Espressif Systems" 3.x - tested with 3.3.12)
------------------------------------------------------------------------------------------------------
    PC first (folder Q250_Hardware_Throw_Fly):   py .\export_ppo_esp32.py      (writes ppo_weights.h here)
    Tools menu: Board "ESP32S3 Dev Module" | USB CDC On Boot "Enabled" | USB Mode "Hardware CDC and JTAG"
                Flash Size "4MB (32Mb)" | Partition Scheme "Default 4MB with spiffs (1.2MB APP/1.5MB SPIFFS)"
                CPU Frequency "240MHz (WiFi)"   ->  Upload  ->  Serial Monitor 115200, line ending "Newline"
    Field: no laptop. It starts by itself when the battery goes in (< 1 s), then waits for GPS.

2. EXPECTED RESULT  (Serial Monitor; the same lines go into console.log on the board)
------------------------------------------------------------------------------------------------------
    | Line                                                            | Means                          |
    |-----------------------------------------------------------------|--------------------------------|
    | boot 3 | mode ppo-policy | files: 2 | free flash 1300 KB        | board up, power-up number 3    |
    | [link] heartbeat from system 1                                  | wiring + SERIAL2 OK            |
    | waiting for the EKF position (GPS fix) ...  (indoors: repeats)  | normal until GPS               |
    | Connected | ppo-policy | hover thrust 0.149 | MOT_SPIN_ARM 0.0..| parameters read                |
    | ESP32 READY TO USE (ppo-policy: 1 long + 2 short beep)          | = the Pi's start beep          |
    | WAIT | CH5 ... | GPS fix 3 ... | 15.3 V | disarmed ALT_HOLD     | every 2 s                      |
    Then exactly as the Pi: CH9 arm -> CH5 ON -> 3 short beeps -> throw -> catch -> hold 10 s -> LAND.

3. TYPED COMMANDS  (Serial Monitor, only while DISARMED)
------------------------------------------------------------------------------------------------------
    | Command          | Does                                                                         |
    |------------------|------------------------------------------------------------------------------|
    | check            | = python3 throw_real.py --check (link, switch, buzzer, toss test). stop ends  |
    | mode ppo / rule  | = mode.txt: saved on the board, restarts with that mode                       |
    | ls / info        | list the files / free space, power-up number, mode                            |
    | get run0003.csv  | print one file (get_logs_esp32.py on the PC saves them all)                   |
    | getall / rmall   | print all files / delete all files (after you saved them on the PC!)          |

4. OUTPUTS  (on the board's flash; the PC script get_logs_esp32.py copies them to runs\throw_esp32\)
------------------------------------------------------------------------------------------------------
    run<boot>.csv        one line per throw: same columns as the Pi's throw_real_<date>.csv (+ steps_dropped)
    run<boot>_steps.csv  every 50 Hz step: same columns as the Pi's _steps.csv (READY: last 1 s only)
    run<boot>.json       settings of that power-up       console.log   every printed line (like the Pi's)
    The date in the files comes from the GPS (UTC). Steps are kept in RAM during the catch and written
    during the GUIDED hold / LAND, because writing the flash pauses the chip for up to ~50 ms.
*/
#include <Arduino.h>
#include <LittleFS.h>
#include <Preferences.h>
#include "hal.h"
#include "throw_core.h"

static const int FC_RX_PIN = 4, FC_TX_PIN = 5;
static const uint32_t FC_BAUD = 921600;
static const size_t CONSOLE_MAX = 256 * 1024;        /* console.log: start a new one above this size */

class EspHal : public Hal {
public:
    File files[4];
    String in, con;                                    /* typed line, console lines not yet on the flash */
    uint32_t con_flush_ms = 0;
    bool fs_ok = false;
    uint32_t ms() override { return millis(); }
    uint32_t us() override { return micros(); }
    void sleep_ms(uint32_t m) override { delay(m); }
    int fc_read(uint8_t *b, int max) override
    {
        int n = Serial1.available();
        if (n <= 0) return 0;
        return Serial1.readBytes(b, n < max ? n : max);
    }
    void fc_write(const uint8_t *b, int n) override { Serial1.write(b, n); }
    void out(const char *line) override
    {
        if (Serial) { Serial.println(line); }          /* no laptop: nothing waits (tx timeout 0) */
        if (con.length() < 8192) { con += line; con += '\n'; }
    }
    void flush_console()                               /* only from the WAIT loop: never during a catch */
    {
        if (!fs_ok || con.length() == 0) return;
        File f = LittleFS.open("/console.log", "a");
        if (f && f.size() > CONSOLE_MAX) {
            f.close();
            LittleFS.remove("/console_old.log");
            LittleFS.rename("/console.log", "/console_old.log");
            f = LittleFS.open("/console.log", "a");
        }
        if (f) { f.print(con); f.close(); }
        con = "";
    }
    bool command(char *buf, int max) override
    {
        if (millis() - con_flush_ms > 1000) { flush_console(); con_flush_ms = millis(); }
        while (Serial && Serial.available()) {
            char c = Serial.read();
            if (c == '\r') continue;
            if (c == '\n') {
                in.trim();
                snprintf(buf, max, "%s", in.c_str());
                in = "";
                return buf[0] != 0;
            }
            if (in.length() < 120) in += c;
        }
        return false;
    }
    int file_open(const char *name, bool append) override
    {
        if (!fs_ok) return -1;
        for (int i = 0; i < 4; i++)
            if (!files[i]) {
                files[i] = LittleFS.open(String("/") + name, append ? "a" : "w");
                return files[i] ? i : -1;
            }
        return -1;
    }
    void file_write(int h, const char *t) override { if (h >= 0 && h < 4 && files[h]) files[h].print(t); }
    void file_close(int h) override { if (h >= 0 && h < 4 && files[h]) files[h].close(); }
    void list(bool print_all)
    {
        flush_console();
        File root = LittleFS.open("/");
        int n = 0;
        for (File f = root.openNextFile(); f; f = root.openNextFile()) {
            if (print_all) print_file(f.name());
            else Serial.printf("%-24s %8u B\n", f.name(), (unsigned)f.size());
            n++;
        }
        Serial.printf("%d files, free flash %u KB\n", n, (unsigned)((LittleFS.totalBytes() - LittleFS.usedBytes()) / 1024));
    }
    void print_file(const char *name)
    {
        File f = LittleFS.open(String("/") + name, "r");
        if (!f) { Serial.printf("no file %s\n", name); return; }
        Serial.printf("=== BEGIN %s %u\n", name, (unsigned)f.size());
        uint8_t b[512];
        while (f.available()) { int n = f.read(b, sizeof(b)); Serial.write(b, n); }
        Serial.printf("\n=== END %s\n", name);
        f.close();
    }
    void file_command(const char *cmd) override
    {
#if ARDUINO_USB_CDC_ON_BOOT
        Serial.setTxTimeoutMs(500);                    /* a file transfer must not lose bytes: wait for the PC */
#endif
        file_command2(cmd);
#if ARDUINO_USB_CDC_ON_BOOT
        Serial.setTxTimeoutMs(0);                      /* back to: never wait (flying) */
#endif
    }
    void file_command2(const char *cmd)
    {
        if (!fs_ok) { Serial.println("no flash file system"); return; }
        if (!strcmp(cmd, "ls")) list(false);
        else if (!strcmp(cmd, "getall")) list(true);
        else if (!strncmp(cmd, "get ", 4)) { flush_console(); print_file(cmd + 4); }
        else if (!strcmp(cmd, "rmall")) {
            File root = LittleFS.open("/");
            String names[64];
            int n = 0;
            for (File f = root.openNextFile(); f && n < 64; f = root.openNextFile()) names[n++] = String("/") + f.name();
            for (int i = 0; i < n; i++) LittleFS.remove(names[i]);
            Serial.printf("deleted %d files\n", n);
        } else if (!strcmp(cmd, "info")) {
            Preferences p;
            p.begin("throw", true);
            Serial.printf("boot %u | mode %s | free flash %u KB | free RAM %u B\n", p.getUInt("boot", 0),
                          p.getBool("ppo", true) ? "ppo-policy" : "attitude-rule",
                          (unsigned)((LittleFS.totalBytes() - LittleFS.usedBytes()) / 1024), (unsigned)ESP.getFreeHeap());
            p.end();
        } else Serial.printf("unknown command '%s' - type help\n", cmd);
    }
    void save_mode(bool ppo) override
    {
        Preferences p;
        p.begin("throw", false);
        p.putBool("ppo", ppo);
        p.end();
    }
    void restart() override
    {
        flush_console();
        delay(5000);
        ESP.restart();
    }
};

static EspHal hal;

void setup()
{
    Serial.begin(115200);
#if ARDUINO_USB_CDC_ON_BOOT
    Serial.setTxTimeoutMs(0);                          /* USB: never wait for a laptop that is not there */
#endif
    Serial1.setRxBufferSize(8192);                     /* ~0.5 s of FC data while the flash is written */
    Serial1.begin(FC_BAUD, SERIAL_8N1, FC_RX_PIN, FC_TX_PIN);
    hal.fs_ok = LittleFS.begin(true);                  /* true: format the first time */
    delay(300);
}

void loop()
{
    Preferences p;
    p.begin("throw", false);
    uint32_t boot = p.getUInt("boot", 0) + 1;
    p.putUInt("boot", boot);
    bool ppo = p.getBool("ppo", true);                 /* default ppo-policy (your choice 8 Nov) */
    p.end();
    char line[120];
    unsigned files = 0;
    if (hal.fs_ok) {
        File root = LittleFS.open("/");
        for (File f = root.openNextFile(); f; f = root.openNextFile()) files++;
    }
    snprintf(line, sizeof(line), "boot %u | mode %s | files: %u | free flash %u KB", (unsigned)boot,
             ppo ? "ppo-policy" : "attitude-rule", files,
             hal.fs_ok ? (unsigned)((LittleFS.totalBytes() - LittleFS.usedBytes()) / 1024) : 0);
    hal.out(line);
    if (!hal.fs_ok) hal.out("WARNING: flash file system not mounted - results will NOT be saved");
    CoreConfig cfg{ppo, boot};
    core_run(hal, cfg);                                /* returns only on a fatal error or mode change */
    hal.out("restarting in 5 s");
    hal.restart();
}
