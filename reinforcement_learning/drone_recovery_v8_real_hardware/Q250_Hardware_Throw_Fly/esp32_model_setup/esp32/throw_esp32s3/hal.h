/* hal.h - what the throw logic (throw_core.cpp) needs from the board.
   ESP32-S3: implemented in throw_esp32s3.ino (UART to the FC, USB console, LittleFS files, NVS settings).
   The same throw_core.cpp is also built on a PC against the ArduPilot simulator (SITL) to test it. */
#pragma once
#include <stdint.h>
#include <stddef.h>

class Hal {
public:
    virtual ~Hal() {}
    virtual uint32_t ms() = 0;                                  /* time since boot, ms */
    virtual uint32_t us() = 0;                                  /* time since boot, us (wraps) */
    virtual void sleep_ms(uint32_t ms) = 0;                     /* let the CPU rest (and the watchdog) */
    virtual int fc_read(uint8_t *buf, int max) = 0;             /* bytes from the FC, never blocks */
    virtual void fc_write(const uint8_t *buf, int n) = 0;       /* bytes to the FC */
    virtual void out(const char *line) = 0;                     /* one printout line (console + console.log) */
    virtual bool command(char *buf, int max) = 0;               /* one typed command line, if any */
    virtual int file_open(const char *name, bool append) = 0;   /* returns handle >= 0, -1 = failed */
    virtual void file_write(int h, const char *text) = 0;
    virtual void file_close(int h) = 0;                         /* also makes it safe on the flash */
    virtual void file_command(const char *cmd) = 0;             /* ls / get / getall / rmall */
    virtual void save_mode(bool ppo) = 0;                       /* remember the mode (like mode.txt) */
    virtual void restart() = 0;                                 /* fatal error: start again after 5 s */
};
