/* throw_core.h - the throw catcher (port of throw_real.py version d) for the ESP32-S3.
   Same rules, numbers, printouts and file columns as throw_real.py; see throw_esp32s3.ino for how to run. */
#pragma once
#include "hal.h"

struct CoreConfig {
    bool ppo;            /* true = ppo-policy (default), false = attitude-rule  (like mode.txt) */
    uint32_t boot_no;    /* counts power-ups: names the result files run<boot_no>... */
};

/* Runs until a fatal error (no FC, no GPS after 2 min, ...). The caller then restarts (like systemd). */
void core_run(Hal &h, const CoreConfig &cfg);
