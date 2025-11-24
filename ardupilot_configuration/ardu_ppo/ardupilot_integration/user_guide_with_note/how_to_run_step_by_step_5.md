🚀 STEP-BY-STEP EXECUTION GUIDE
📍 WHERE YOU ARE NOW
You've successfully completed:

✅ Step 1: Setup
✅ Step 2: Start SITL
✅ Step 3: Connection test
✅ Step 4: Stage 1 hover

🎯 STEP 5: DEPLOY STAGE 2 (WIND DISTURBANCE)
Preparation (Terminal 1 - SITL console):

# Enable wind in SITL
param set SIM_WIND_SPD 5.0
param set SIM_WIND_DIR 180
param set SIM_WIND_TURB 1.0
param set SIM_WIND_T 1
```

**Expected SITL response:**
```
SIM_WIND_SPD = 5.000000
SIM_WIND_DIR = 180.000000
SIM_WIND_TURB = 1.000000
SIM_WIND_T = 1.000000