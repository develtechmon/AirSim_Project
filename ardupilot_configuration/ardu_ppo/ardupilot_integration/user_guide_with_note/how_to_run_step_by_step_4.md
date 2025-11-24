🚁 STEP 4: DEPLOY STAGE 1 (HOVER)
Terminal 2:

python 3_deploy_stage1_hover.py --duration 30
```

**Expected output (watch this closely!):**
```
======================================================================
🚁 DEPLOYING STAGE 1: HOVER POLICY
======================================================================
Model: ../models/hover_policy_best.pth
Target altitude: 10.0m
Duration: 30s
Control rate: 20Hz
======================================================================

[1/4] Connecting to drone...
✅ Connected to ArduPilot
   Home: Lat=-35.363262, Lon=149.165237, Alt=0.00m

[2/4] Loading trained model...
Loading Stage 1 model: ../models/hover_policy_best.pth
✅ Stage 1 model loaded
   Architecture: 13 → 256 → 256 → 128 → 3
   Outputs: [vx, vy, vz] velocity commands

[3/4] Taking off...
✅ API control enabled (GUIDED mode)
Waiting for vehicle to be armable...
Arming...
✅ Armed
Taking off to 10.0m...
  Altitude: 1.2m / 10.0m
  Altitude: 3.5m / 10.0m
  Altitude: 5.8m / 10.0m
  Altitude: 8.1m / 10.0m
  Altitude: 9.7m / 10.0m
✅ Target altitude reached
✅ At 10.0m altitude

[4/4] Deploying hover policy...
======================================================================
🎯 HOVER CONTROL ACTIVE
======================================================================
Press Ctrl+C to stop

[   0.0s] Altitude: 10.02m (error: 0.02m) | Distance: 0.15m | Action: [-0.05,  0.12, -0.03]
[   2.0s] Altitude:  9.98m (error: 0.02m) | Distance: 0.23m | Action: [ 0.08, -0.06,  0.01]
[   4.0s] Altitude: 10.01m (error: 0.01m) | Distance: 0.18m | Action: [-0.03,  0.04, -0.02]
[   6.0s] Altitude:  9.99m (error: 0.01m) | Distance: 0.21m | Action: [ 0.02, -0.01,  0.00]
[   8.0s] Altitude: 10.00m (error: 0.00m) | Distance: 0.19m | Action: [ 0.01,  0.03, -0.01]
[  10.0s] Altitude: 10.02m (error: 0.02m) | Distance: 0.17m | Action: [-0.04,  0.02,  0.01]
[  12.0s] Altitude:  9.97m (error: 0.03m) | Distance: 0.22m | Action: [ 0.06, -0.03,  0.02]
[  14.0s] Altitude: 10.01m (error: 0.01m) | Distance: 0.16m | Action: [-0.02,  0.01, -0.01]
[  16.0s] Altitude:  9.99m (error: 0.01m) | Distance: 0.20m | Action: [ 0.03, -0.02,  0.00]
[  18.0s] Altitude: 10.00m (error: 0.00m) | Distance: 0.18m | Action: [ 0.01,  0.01,  0.00]
[  20.0s] Altitude: 10.02m (error: 0.02m) | Distance: 0.21m | Action: [-0.05,  0.02,  0.01]
[  22.0s] Altitude:  9.98m (error: 0.02m) | Distance: 0.19m | Action: [ 0.04, -0.01,  0.00]
[  24.0s] Altitude: 10.01m (error: 0.01m) | Distance: 0.17m | Action: [-0.02,  0.03, -0.01]
[  26.0s] Altitude:  9.99m (error: 0.01m) | Distance: 0.20m | Action: [ 0.03, -0.02,  0.01]
[  28.0s] Altitude: 10.00m (error: 0.00m) | Distance: 0.18m | Action: [ 0.01,  0.01,  0.00]

======================================================================
📊 HOVER PERFORMANCE STATISTICS
======================================================================
Altitude:
  Mean: 10.005m (target: 10.0m)
  Std dev: 0.098m
  Max error: 0.287m

Position:
  Mean distance: 0.192m
  Max distance: 0.423m

Control:
  Total steps: 600
  Control rate: 20.0 Hz

======================================================================
✅ EXCELLENT! Stable hover maintained
======================================================================

Landing...
✅ API control disabled (STABILIZE mode)
✅ Disarmed
✅ Stage 1 deployment complete!
```

---

## 🎯 **SUMMARY: WHICH FILE TO RUN FIRST**

### **Order of Execution:**
```
1️⃣ START: sim_vehicle.py (Terminal 1 - keep running)
   └─ Wait for "Ready to fly!"

2️⃣ TEST: python 1_connection_test.py
   └─ Look for "✅ Connection test complete!"

3️⃣ STAGE 1: python 3_deploy_stage1_hover.py --duration 30
   └─ Look for "✅ EXCELLENT! Stable hover maintained"

4️⃣ (Later) STAGE 2 & 3 scripts