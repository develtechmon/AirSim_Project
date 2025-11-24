STEP 2: START ARDUPILOT SITL
Terminal 1 (keep this open):

cd ~/ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter --console --map
```

**Expected output:**
```
Init ArduCopter V4.3.0
EKF2 IMU0 initialised
GPS 1: detected GPS type 1
EKF2 IMU1 initialised
PreArm: EKF2 is using GPS
APM: ArduCopter V4.3.0 (e58a7d6b)
Mode: STABILIZE
Armed: False