Now run comprehensive tests:

# Test 1: Low intensity (3 tests)
python 5_deploy_stage3_recovery.py --intensity 0.5 --tests 3

# Test 2: Medium intensity (3 tests)
python 5_deploy_stage3_recovery.py --intensity 0.7 --tests 3

# Test 3: High intensity (3 tests)
python 5_deploy_stage3_recovery.py --intensity 1.0 --tests 3

# Test 4: Different disturbance types
python 5_deploy_stage3_recovery.py --type flip --intensity 0.5 --tests 2
python 5_deploy_stage3_recovery.py --type spin --intensity 0.5 --tests 2

# Test 5: PhD requirement - 20+ tests total
python 5_deploy_stage3_recovery.py --intensity 0.7 --tests 10
```

---

## 📈 **COLLECTING PHD DATA**

After running all tests, you'll have:
```
✅ Stage 1: Hover stability data
✅ Stage 2: Wind disturbance handling data
✅ Stage 3: Impact recovery data
   - Success rates at different intensities
   - Recovery times
   - Altitude profiles
   - Angular velocity profiles
```

**Create summary table:**
```
| Intensity | Tests | Success | Rate | Avg Time | Min Alt | Max AngVel |
|-----------|-------|---------|------|----------|---------|------------|
| 0.5x      | 3     | 3       | 100% | 10.2s    | 25.1m   | 2.34rad/s  |
| 0.7x      | 3     | 3       | 100% | 11.5s    | 23.4m   | 3.12rad/s  |
| 1.0x      | 3     | 2       | 67%  | 13.8s    | 21.2m   | 4.56rad/s  |
```

---

## ✅ **CHECKLIST: FILES YOU HAVE NOW**
```
phd_ardupilot_deploy/
├── ardupilot_integration/
│   ├── utils/
│   │   ├── ardupilot_interface.py       ✅
│   │   └── model_loader.py              ✅
│   ├── 1_connection_test.py             ✅
│   ├── 3_deploy_stage1_hover.py         ✅
│   ├── 4_deploy_stage2_disturbance.py   ✅
│   └── 5_deploy_stage3_recovery.py      ✅
└── models/
    ├── hover_policy_best.pth
    ├── hover_disturbance_policy.zip
    └── stage3_checkpoints/
        ├── gated_curriculum_policy.zip
        └── gated_curriculum_vecnormalize.pkl