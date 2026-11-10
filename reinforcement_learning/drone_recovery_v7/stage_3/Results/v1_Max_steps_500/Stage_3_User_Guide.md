# Stage 3 — Performance-Gated Curriculum Impact Recovery
## Complete User Guide

**Project:** Three-Stage Curriculum Reinforcement Learning for Impact-Resilient Quadrotor UAV Attitude Recovery in AirSim
**Author:** Lukas Anak Johnny — PhD, UiTM Shah Alam
**Environment:** Windows 10/11, AirSim SimpleFlight, Python 3.11

---

## Prerequisites

Before running Stage 3, confirm all of the following are ready:

| Requirement | Detail |
|---|---|
| Stage 2 model | `./models/hover_disturbance_policy.zip` must exist |
| AirSim running | Launch Unreal Engine with AirSim before any script |
| Python environment | `airsimcode` conda environment activated |
| Working directory | `C:\Users\lukas\Desktop\My_Project\AirSim_Project\reinforcement_learning\drone_recovery_v7` |

Activate your environment before starting:

```
conda activate airsimcode
```

---

## Overview — Three Scripts, Three Steps

| Step | Script | Purpose | Duration |
|---|---|---|---|
| 1 | `train_stage_3_gated_curriculum.py` | Train the gated curriculum policy | ~14 hours |
| 2 | `analyze_training_logs.py` | Analyse training logs and generate plots | ~1 minute |
| 3 | `test_stage_3_gated_curriculum.py` | Evaluate final policy on 60 test episodes | ~30 minutes |

**Important:** Always run in this exact order. Step 2 and Step 3 both depend on outputs from Step 1.

---

## Step 1 — Train the Gated Curriculum Policy

### Command

```
py .\stage_3\train_stage_3_gated_curriculum.py --timesteps 600000
```

### What This Script Does

- Loads Stage 2 weights from `./models/hover_disturbance_policy.zip` into the PPO actor network
- Trains the policy under three physical disturbance types: Bird Attack, Flip, and Spin
- Advances curriculum automatically through three levels — Easy, Medium, Hard — based on performance gating
- Saves model checkpoints every 50,000 steps and at each curriculum level transition
- Writes a per-episode training log to `./logs/stage3/`

### Curriculum Levels

| Level | Intensity Range | Advance Condition |
|---|---|---|
| Easy | 0.7–0.9× | ≥ 80% rolling 50-episode recovery rate AND minimum 50 episodes |
| Medium | 0.9–1.1× | ≥ 70% rolling 50-episode recovery rate AND minimum 50 episodes |
| Hard | 1.1–1.5× | Final level — trains until timestep limit reached |

### Console Output During Training

The script prints a progress line every 10 episodes. You will see something like:

```
Episode   10 | Level: EASY   | Reward: 21304.2 | Recovery: 100.0% | Intensity: 0.81x
Episode   20 | Level: EASY   | Reward: 22871.5 | Recovery: 100.0% | Intensity: 0.77x
...
Episode   51 | Level: MEDIUM | Reward: 21363.0 | Recovery: 100.0% | Intensity: 1.01x
[CURRICULUM] Advanced to MEDIUM at episode 51 (0.7h)
...
Episode  101 | Level: HARD   | Reward: 21663.0 | Recovery:  98.0% | Intensity: 1.14x
[CURRICULUM] Advanced to HARD at episode 101 (1.4h)
...
Episode 1002 | Level: HARD   | Reward: 33191.7 | Recovery: 100.0% | Intensity: 1.30x
```

### Expected Curriculum Progression

| Event | Episode | Time |
|---|---|---|
| Training begins at Easy | Episode 1 | 0.0h |
| Advances to Medium | Episode 51 | 0.7h |
| Advances to Hard | Episode 101 | 1.4h |
| Training completes | Episode 1002 | 14.0h |

### Output Files Produced

```
./logs/stage3/gated_training_YYYYMMDD_HHMMSS_episodes.csv
./logs/stage3/gated_training_YYYYMMDD_HHMMSS_summary.json
./logs/stage3/gated_training_YYYYMMDD_HHMMSS_curriculum.json

./models/stage3_checkpoints/gated_curriculum_policy.zip          ← final model
./models/stage3_checkpoints/gated_curriculum_vecnormalize.pkl
./models/stage3_checkpoints/curriculum_levels/level_0_EASY_mastered.zip
./models/stage3_checkpoints/curriculum_levels/level_1_MEDIUM_mastered.zip
./models/stage3_checkpoints/gated_checkpoints/gated_curriculum_50000_steps.zip
./models/stage3_checkpoints/gated_checkpoints/gated_curriculum_550000_steps.zip
./models/stage3_checkpoints/gated_checkpoints/gated_curriculum_600000_steps.zip
```

### Expected Final Training Summary

When training completes you will see:

```
======================================================================
✅ GATED TRAINING COMPLETE!
======================================================================

📊 Training Statistics:
   Total episodes: 1002
   Avg return: 32946.6 (last 50)
   Recovery rate: 100% (last 50)
   Avg recovery time: 18 steps

🎓 Curriculum Progression:
   Level 0 (EASY  0.7-0.9): Reached at episode   1 (0.0h)
   Level 1 (MEDIUM 0.9-1.1): Reached at episode  51 (0.7h)
   Level 2 (HARD  1.1-1.5): Reached at episode 101 (1.4h)
```

### What Good Training Looks Like

The reward curve follows a U-shape — this is expected and correct:

- **Start (ep 1–50, Easy):** Reward strongly positive ~21,000–22,000 from Stage 2 warm-start
- **Dip (ep 101–300, early Hard):** Reward declines to a minimum of ~18,681 as the curriculum pushes the policy beyond its current capability
- **Recovery (ep 300–1002, late Hard):** Reward climbs consistently to ~31,000–33,000 as the policy adapts
- **Critical check:** Reward should never drop below zero at any point. If it does, the Stage 2 weights may not have loaded correctly.

---

## Step 2 — Analyse Training Logs

### Command

Replace `YYYYMMDD_HHMMSS` with the actual timestamp from your Step 1 log filename.

```
py .\stage_3\analyze_training_logs.py --log .\logs\stage3\gated_training_YYYYMMDD_HHMMSS_episodes.csv
```

**Example using the confirmed filename from this project:**

```
py .\stage_3\analyze_training_logs.py --log .\logs\stage3\gated_training_20260808_111241_episodes.csv
```

### What This Script Does

- Reads the per-episode CSV log from Step 1
- Prints a full training statistics summary to console
- Generates and saves four training plots
- Saves LaTeX-formatted tables for thesis use

### Expected Console Output

```
📂 Loading training data from: .\logs\stage3\gated_training_20260808_111241_episodes.csv
   ✅ Loaded 1002 episodes

📈 Generating learning curves...
   ✅ Saved: plots\training_learning_curves.png
   ✅ Saved: plots\training_learning_curves.pdf

📊 Generating curriculum transition plot...

📈 Curriculum Transitions:
   Level 0 (EASY)   → Level 1 (MEDIUM)  At episode  51 (0.7h)
   Level 1 (MEDIUM) → Level 2 (HARD)    At episode 101 (1.4h)
   ✅ Saved: plots\curriculum_progression.png
   ✅ Saved: plots\curriculum_progression.pdf

================================================================================
📊 TRAINING STATISTICS SUMMARY
================================================================================

OVERALL PERFORMANCE
  Total Episodes:      1002
  Total Training Time: 14.0 hours
  Final Recovery Rate: 100.0% (last 50)
  Final Avg Reward:    33191.7 (last 10)

CURRICULUM PROGRESSION
Level Transition    Episode    Time (hours)    Episodes Trained
Level 0 → 1          51          0.7              50
Level 1 → 2         101          1.4              50
Level 2 (Final)      101         14.0             902

BY CURRICULUM LEVEL
Level               Episodes    Recovery Rate    Avg Intensity
EASY  (0.7-0.9×)      50          100.0%           0.79×
MEDIUM (0.9-1.1×)     50          100.0%           1.02×
HARD  (1.1-1.5×)     902           99.7%           1.30×

📄 LaTeX tables saved:
   - analysis_results/table_5_1_training_statistics.tex
   - analysis_results/table_5_2_curriculum_advancement.tex

✅ Statistics saved to: plots\training_statistics.json
🎉 Analysis complete! Plots saved to: ./plots/
```

### Output Files Produced

```
./plots/training_learning_curves.png    ← 4-panel training plot for paper/thesis
./plots/training_learning_curves.pdf
./plots/curriculum_progression.png     ← rolling recovery rate with level shading
./plots/curriculum_progression.pdf
./plots/training_statistics.json
./analysis_results/table_5_1_training_statistics.tex
./analysis_results/table_5_2_curriculum_advancement.tex
```

### What the Plots Show

**training_learning_curves.png** — 4-panel figure:
- Top-left: Episode reward over time with rolling 10-episode average
- Top-right: Rolling 50-episode recovery rate with level threshold lines at 80%, 70%, 60%
- Bottom-left: Curriculum level step function (Easy → Medium → Hard)
- Bottom-right: Disturbance intensity scatter coloured by recovery outcome (green = recovered, red = failed)

**curriculum_progression.png** — standalone figure:
- X-axis: Training episode 1–1002
- Y-axis: Recovery rate (%)
- Background shading: green (Easy ep 1–50), yellow (Medium ep 51–100), red (Hard ep 101–1002)
- Horizontal dashed lines: 80% / 70% / 60% threshold targets
- Rolling 50-episode recovery rate as dark blue line

---

## Step 3 — Test the Final Policy

### Command

```
py .\stage_3\test_stage_3_gated_curriculum.py --episodes 60
```

### What This Script Does

- Loads the final trained model from `./models/stage3_checkpoints/gated_curriculum_policy.zip`
- Runs 60 independent test episodes at Hard level (intensity 1.1–1.5×)
- Prints per-episode results and a full summary
- Reports performance against the PhD target of ≥60% Hard-level recovery rate

### Expected Console Output

```
======================================================================
📊 COMPREHENSIVE TEST RESULTS
======================================================================

📈 Overall Performance:
   Success Rate: 100% (60/60 episodes)
   Average Distance: 0.16m (successful episodes)
   Average Episode Length: 600.0 steps

🔄 Overall Recovery:
   Episodes with disturbance: 60/60
   Successful recoveries:     60/60
   Overall Recovery Rate:     100%
   Avg Recovery Time:         19 steps (0.9s)

======================================================================
🎯 PERFORMANCE BY CURRICULUM LEVEL (PhD Analysis)
======================================================================

   Level 2 - HARD (1.1-1.5):
      Recovery Rate: 100.0% (60/60) ✅ MASTERED
      Target: 60%
      Avg Intensity: 1.30×

🔬 DETAILED INTENSITY BREAKDOWN:
   1.1-1.3× (HARD):    100.0% (33/33)
   1.3-1.5× (EXTREME): 100.0% (27/27)

🎯 Performance Grade: A+ (OUTSTANDING)

🎓 PhD GATED CURRICULUM ASSESSMENT:
   ✅ Level 2 (HARD 1.1-1.5): MASTERED (100% ≥ 60%)
   🎉 ALL LEVELS MASTERED!
   ✅ PhD requirements met
   ✅ Performance-gated curriculum validated
======================================================================
```

### Pass Criteria

| Metric | PhD Target | Confirmed Result |
|---|---|---|
| Hard-level recovery rate | ≥ 60% | 100% (60/60) ✅ |
| Avg 3D distance from target | — | 0.16m ✅ |
| Avg recovery time | — | 19 steps (0.9s) ✅ |

---

## Complete Results Summary — All Three Stages

| Metric | Stage 1 BC | Stage 2 PPO Wind | Stage 3 Curriculum |
|---|---|---|---|
| Success rate | 100% (10/10) | 100% (10/10) | 100% (60/60) |
| Avg 3D distance | 0.03m | 0.80m | 0.16m |
| Max disturbance handled | None — hover only | 4.9 m/s wind | 1350 deg/s impact |
| Training time | 8.7 min | 12 hours | 14 hours |
| Training episodes | — | 1,003 | 1,002 |

---

## Troubleshooting

**AirSim connection refused at startup**
Make sure Unreal Engine with AirSim is fully loaded and showing the drone before running any script. Wait for the AirSim window to finish loading before executing the command.

**Script exits immediately with model not found error**
Confirm `./models/hover_disturbance_policy.zip` exists. Stage 3 loads Stage 2 weights at startup. If the file is missing, re-run Stage 2 training first.

**Reward drops to large negative values early in training**
This means the Stage 2 weights did not load correctly and the policy started from random initialisation. Stop training, verify the model path, and restart Step 1.

**Curriculum does not advance from Easy after 50 episodes**
The rolling 50-episode recovery rate did not reach 80%. This should not happen with correct Stage 2 initialisation. Check that AirSim is running in SimpleFlight mode and not PX4 mode.

**analyze_training_logs.py — file not found error**
The timestamp in your command does not match the actual log filename. Check `.\logs\stage3\` for the correct filename and replace `YYYYMMDD_HHMMSS` with the actual timestamp.

**test_stage_3_gated_curriculum.py — vecnormalize error**
The test script requires both `gated_curriculum_policy.zip` AND `gated_curriculum_vecnormalize.pkl` to be in `./models/stage3_checkpoints/`. Both are saved at the end of Step 1.