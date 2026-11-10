# Stage 1 Summary: Hover Training (ArduPilot SITL + AirSim) - RUN1

**Run folder:** `runs\run1`  
**Generated:** 2026-10-23 09:54  
**Sensor mode:** gps  
**Firmware:** 4.4.0-dev  
**Method:** Behavioural cloning of the expert (observable P expert: v = gain x (target - position); xy gain 0.8/s cap 1.0 m/s, z gain 1.0/s cap 0.5 m/s)  
**Observation:** 13 (position minus target, velocity, quaternion, body rates)  
**Action:** 3 velocity commands (vx, vy, vz), GUIDED mode

All numbers below are read from the saved result files. Each status follows the rule in its row.

## Executive summary

| Step | Script | Status | Key result | Pass rule |
| --- | --- | --- | --- | --- |
| 1. Data collection | `collect_demonstration.py` | **PASS** | 60,000 samples, loop 50.0 ms, expert settled 100% | loop within 5% of target, overruns <= 1%, expert settled >= 80% |
| 2. Training | `train_imitation.py` | **PASS** | best val loss 0.0002 (epoch 148), transfer diff 0.0e+00 | transfer to PPO identical (diff < 1e-5) |
| 3. Test (network alone) | `test_hover_policy.py` | **NOT RUN** | - | - |
| 4. PID vs BC | `collect_stage1_statistics.py` | **PASS** | transfer efficiency 98.9% (position), 97.3% (altitude) | transfer efficiency >= 90% |

## Overall performance

```
Position RMSE PID/BC:  0.344 / 0.348 m
Transfer efficiency:   98.9% (position), 97.3% (altitude)
Best validation loss:  0.0002 (epoch 148)
Dataset:               60,000 samples, 200 episodes
```

## Step 1: Data collection (`collect_demonstration.py`)

Command: `py .\collect_demonstration.py --episodes 200 --steps 300`

| Item | Value |
| --- | --- |
| File | `expert_demonstrations_gps_20261017_170846.pkl` |
| Samples / episodes | 60,000 / 200 |
| Control rate | 20 Hz target, 50.0 ms measured (max 65.9 ms, 0 overruns) |
| Heights | 1.0-3.0 m, targets +/-1.0 m xy, +/-0.5 m z |
| PID expert settled | 100% (end error < 0.3 m), mean end error 0.011 m |
| Safety stop | none |

## Step 2: Training (`train_imitation.py`)

| Item | Value |
| --- | --- |
| Architecture | 13-256-256-128-3, ReLU, 102,659 parameters |
| Split | by episode 90/10: 54,000 train / 6,000 validation |
| Settings | 150 epochs, batch 256, lr 0.001, seed 0 |
| Best validation loss | 0.0002 at epoch 148 (saved model) |
| Validation RMSE per axis | vx 0.023, vy 0.006, vz 0.005 m/s |
| Transfer to PPO | max action difference 0.0 |
| Loss log | `runs\run1\logs\stage1\training_20261017_180044.csv` |

Loss at selected epochs:

| Epoch | Train loss | Val loss |
| --- | --- | --- |
| 1 | 0.0010 | 0.0005 |
| 10 | 0.0001 | 0.0003 |
| 25 | 0.0000 | 0.0003 |
| 50 | 0.0000 | 0.0003 |
| 100 | 0.0000 | 0.0002 |
| 148 (best) | 0.0000 | 0.0002 |
| 150 | 0.0000 | 0.0002 |

## Step 4: PID vs BC (`collect_stage1_statistics.py`)

Same 10 targets for both (seed 42). Source: `statistics_summary_20261017_191030.json`

| Metric | PID expert | BC policy | Gap |
| --- | --- | --- | --- |
| Position error RMSE (m) | 0.344 | 0.348 | +0.004 |
| Altitude RMSE (m) | 0.083 | 0.085 | +0.002 |
| Max horizontal distance (m) | 1.103 | 1.102 | -0.001 |
| End error (m) | 0.011 | 0.036 | +0.025 |
| Success rate (%) | 100.000 | 100.000 | +0.000 |
| Settled rate (%) | 100.000 | 100.000 | +0.000 |
| Transfer efficiency, position | | 98.9% | |
| Transfer efficiency, altitude | | 97.3% | |

Transfer efficiency = (1 - (BC_RMSE - PID_RMSE) / PID_RMSE) x 100%.

## Target vs achieved

| Target | Rule | Achieved | Status |
| --- | --- | --- | --- |
| Transfer to Stage 2 works | action diff < 1e-5 | 0.0 | PASS |
| Network matches the PID | transfer efficiency >= 90% | 98.9% | PASS |
| Fixed control rate | within 5% of 50 ms | 50.0 ms | PASS |

## Files

| File | Use |
| --- | --- |
| `runs\run1\demonstrations\expert_demonstrations_gps_20261017_170846.pkl` | training data |
| `runs\run1\models\hover_policy_best.pth` | trained network |
| `runs\run1\models\obs_norm.npz` | input scaling (Stage 2 reuses it) |
| `runs\run1\models\stage1_ppo_init.zip` | Stage 2 starting point |
| `runs\run1\logs\stage1\training_20261017_180044.csv` | loss curve data (Graph 1) |
| `-` | test success rate |
| `runs\run1\results\statistics_summary_20261017_191030.json` | PID vs BC numbers (Graphs 2-5, RMSE table) |

## Readiness for Stage 2

- [x] Model trained (`hover_policy_best.pth`)
- [x] Stage 2 start file (`stage1_ppo_init.zip`)
- [x] Transfer to PPO verified
- [ ] Test success >= 90%
- [x] Transfer efficiency >= 90%
- [ ] Data collected in optical-flow mode
