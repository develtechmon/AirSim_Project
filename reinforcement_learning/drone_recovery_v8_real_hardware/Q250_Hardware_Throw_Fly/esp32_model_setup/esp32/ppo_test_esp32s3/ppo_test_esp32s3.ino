/*
ppo_test_esp32s3.ino - PHASE 1: does the Round 1 PPO run correctly and fast enough on the ESP32-S3?
=====================================================================================================
1. HOW TO RUN  (Arduino IDE, ESP32-S3 Super Mini on USB)
   PC first: py .\export_ppo_esp32.py   (writes ppo_weights.h + test_vectors.h into this folder)
   Tools: Board "ESP32S3 Dev Module", USB CDC On Boot "Enabled", Flash Size "4MB (32Mb)",
          CPU Frequency "240MHz (WiFi)"  ->  Upload  ->  Serial Monitor 115200 (press RST once if empty)

2. EXPECTED RESULT  (repeats every 5 s)
   | Line                                                         | Means                              |
   |--------------------------------------------------------------|------------------------------------|
   | PPO round1_final.zip on ESP32-S3 @ 240 MHz, free RAM ... B   | model in RAM, board recognised     |
   | correctness: 200 inputs, max difference vs ONNX 0.00004 (OK) | same actions as the Pi (< 0.001)   |
   | speed: xxxx us per step -> OK for 50 Hz (20000 us)           | fast enough (expected ~3000-5000)  |
   "TOO BIG": ppo_weights.h and test_vectors.h are not from the same export -> run the PC script again.
   "TOO SLOW": CPU Frequency is not 240 MHz -> Tools menu.

3. FILES  (all made / used in this folder)
   ppo_mlp.h / ppo_mlp.cpp  the network maths      ppo_weights.h   the numbers (export_ppo_esp32.py)
   test_vectors.h           inputs + ONNX answers (export_ppo_esp32.py)
*/
#include "ppo_mlp.h"
#include "ppo_weights.h"
#include "test_vectors.h"

void setup()
{
    Serial.begin(115200);
    delay(1500);
}

void loop()
{
    Serial.printf("\nPPO %s on ESP32-S3 @ %u MHz, free RAM %u B\n", PPO_SOURCE, (unsigned)getCpuFrequencyMhz(),
                  (unsigned)ESP.getFreeHeap());
    float act[3], worst = 0.0f;
    for (int n = 0; n < PPO_N_TEST; n++) {
        ppo_act(&PPO_TEST_OBS[n * 13], act);
        for (int k = 0; k < 3; k++) {
            float d = fabsf(act[k] - PPO_TEST_ACT[n * 3 + k]);
            if (d > worst) worst = d;
        }
    }
    Serial.printf("correctness: %d inputs, max difference vs ONNX %.5f (%s)\n", PPO_N_TEST, worst,
                  worst < 1e-3f ? "OK" : "TOO BIG");
    uint32_t t0 = micros();
    for (int r = 0; r < 200; r++) ppo_act(&PPO_TEST_OBS[(r % PPO_N_TEST) * 13], act);
    uint32_t us = (micros() - t0) / 200;
    Serial.printf("speed: %u us per step -> %s for 50 Hz (20000 us)\n", (unsigned)us, us < 20000 ? "OK" : "TOO SLOW");
    delay(5000);
}
