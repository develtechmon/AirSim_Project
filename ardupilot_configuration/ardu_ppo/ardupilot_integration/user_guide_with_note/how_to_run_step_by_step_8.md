Run Stage 3 - START WITH LOW INTENSITY! (Terminal 2):

python 5_deploy_stage3_recovery.py --intensity 0.5 --tests 1
```

**Expected Output (DRAMATIC!):**
```
======================================================================
🎯 DEPLOYING STAGE 3: IMPACT RECOVERY
======================================================================
THIS IS YOUR PHD'S MAIN CONTRIBUTION!

Model: ../models/stage3_checkpoints/gated_curriculum_policy.zip
VecNormalize: ../models/stage3_checkpoints/gated_curriculum_vecnormalize.pkl
Target altitude: 30.0m
Disturbance: bird_attack (intensity: 0.5x)
Number of tests: 1
Control rate: 20Hz

⚠️  SAFETY:
   - Ensure sufficient altitude for recovery
   - Have manual RC override ready
   - Start with low intensity (0.5)
======================================================================

[1/5] Connecting to drone...
✅ Connected to ArduPilot
   Home: Lat=-35.363262, Lon=149.165237, Alt=0.00m

[2/5] Loading trained recovery model...
Loading Stage 3 model: ../models/stage3_checkpoints/gated_curriculum_policy.zip
Loading normalization stats: ../models/stage3_checkpoints/gated_curriculum_vecnormalize.pkl
✅ Stage 3 model loaded
   Architecture: 13 → 256 → 256 → 128 → 3
   Outputs: [vx, vy, vz] velocity commands
   Normalization: Enabled

[3/5] Taking off to safe altitude...
✅ API control enabled (GUIDED mode)
Arming...
✅ Armed
Taking off to 30.0m...
  Altitude: 3.4m / 30.0m
  Altitude: 7.2m / 30.0m
  Altitude: 11.5m / 30.0m
  Altitude: 15.8m / 30.0m
  Altitude: 20.1m / 30.0m
  Altitude: 24.3m / 30.0m
  Altitude: 28.6m / 30.0m
  Altitude: 29.8m / 30.0m
✅ Target altitude reached
✅ At 30.0m altitude - SAFE for testing

[4/5] Initial stabilization (10 seconds)...
✅ Stable hover established

[5/5] Running impact recovery tests...
======================================================================
🧪 TESTING BIRD_ATTACK RECOVERY
======================================================================


======================================================================
TEST 1/1
======================================================================

[Phase 1] Pre-disturbance hover (5s)...
✅ Stable at 29.98m


[Phase 2] 🐦 APPLYING BIRD_ATTACK!
   💥 Impact applied!
   Type: bird_attack
   Intensity: 0.5x


[Phase 3] 🚁 AUTONOMOUS RECOVERY IN PROGRESS...
   [ 0.0s] Alt: 29.87m | Dist: 0.34m | AngVel: 0.23rad/s | Upright: ✅
   [ 1.0s] Alt: 28.56m | Dist: 0.89m | AngVel: 1.87rad/s | Upright: ❌  ← TUMBLING!
   [ 2.0s] Alt: 27.23m | Dist: 1.45m | AngVel: 2.34rad/s | Upright: ❌  ← SPINNING!
   [ 3.0s] Alt: 26.12m | Dist: 1.67m | AngVel: 1.56rad/s | Upright: ❌  ← STILL TUMBLING
   [ 4.0s] Alt: 25.45m | Dist: 1.45m | AngVel: 0.89rad/s | Upright: ✅  ← GETTING UPRIGHT!
   [ 5.0s] Alt: 25.12m | Dist: 1.23m | AngVel: 0.45rad/s | Upright: ✅  ← STABILIZING...
   [ 6.0s] Alt: 25.67m | Dist: 0.98m | AngVel: 0.28rad/s | Upright: ✅  ← CLIMBING!
   [ 7.0s] Alt: 26.89m | Dist: 0.76m | AngVel: 0.19rad/s | Upright: ✅  ← CLIMBING MORE!
   [ 8.0s] Alt: 28.12m | Dist: 0.58m | AngVel: 0.12rad/s | Upright: ✅  ← ALMOST THERE!
   [ 9.0s] Alt: 29.23m | Dist: 0.43m | AngVel: 0.07rad/s | Upright: ✅  ← VERY CLOSE!
   [10.0s] Alt: 29.78m | Dist: 0.34m | AngVel: 0.04rad/s | Upright: ✅  ← PERFECT!

   ✅ RECOVERY SUCCESSFUL!
   ⏱️  Recovery time: 10.45s
   📍 Final altitude: 29.81m
   📍 Final distance: 0.32m


[Phase 4] Checking post-recovery stability (5s)...
   ✅ Post-recovery: Alt=29.84m


======================================================================
📊 OVERALL RECOVERY TEST RESULTS
======================================================================

Success Rate: 100% (1/1 tests)
Average Recovery Time: 10.45s

Test Details:
  Test 1: ✅ RECOVERED | Time: 10.45s | Min Alt: 25.12m | Max AngVel: 2.34rad/s

======================================================================
🎓 PhD ASSESSMENT
======================================================================
✅ OUTSTANDING! Your system demonstrates excellent recovery!
   This validates your PhD hypothesis:
   'Impact-resilient UAV can autonomously recover from impacts'
======================================================================

Landing...
✅ Landed
✅ Stage 3 deployment complete!
```

---

### **🎬 WHAT JUST HAPPENED (THE STORY):**
```
00:00-00:05 | Drone hovering peacefully at 30m ✅
            |
00:05       | 💥 BIRD STRIKE! 
            |
00:05-00:07 | 🌀 TUMBLING! Angular velocity: 2.34 rad/s (134°/sec!)
            | ❌ NOT UPRIGHT
            | Altitude dropping: 30m → 27m → 25m
            |
00:07-00:09 | 🤖 Neural network fighting back!
            | Sending strong correction commands
            | ✅ Getting upright again!
            | Spin slowing: 2.34 → 0.89 rad/s
            |
00:09-00:15 | 🚀 CLIMBING BACK!
            | Altitude: 25m → 27m → 29m → 30m
            | Centering position
            | Angular velocity: < 0.1 rad/s (stable!)
            |
00:15+      | ✅ RECOVERED!
            | Stable hover at 30m like nothing happened!
