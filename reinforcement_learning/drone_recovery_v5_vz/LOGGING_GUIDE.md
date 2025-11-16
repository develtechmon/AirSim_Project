# 🔧 TRAINING WITH COMPREHENSIVE LOGGING - CORRECTED PATHS

## ✅ **ALL LOGS SAVE TO: `logs/stage3/`**

---

## 📁 **FILE STRUCTURE:**

```
your_project/
├── train_gated_curriculum_with_logging.py    ← Enhanced training script
├── analyze_training_logs.py                  ← Analysis & plotting script
│
├── logs/
│   └── stage3/                               ← ALL LOGS HERE ✅
│       ├── gated_training_TIMESTAMP_episodes.csv
│       ├── gated_training_TIMESTAMP_summary.json
│       └── gated_training_TIMESTAMP_curriculum.json
│
├── plots/                                    ← Generated plots
│   ├── training_learning_curves.pdf
│   ├── curriculum_progression.pdf
│   └── training_statistics.json
│
└── models/stage3_checkpoints/
    ├── curriculum_levels/
    │   ├── level_0_EASY_mastered.zip
    │   └── level_1_MEDIUM_mastered.zip
    └── gated_curriculum_policy.zip
```

---

## 🚀 **STEP 1: RUN TRAINING WITH LOGGING**

```bash
python train_gated_curriculum_with_logging.py \
    --stage2-model ./models/hover_disturbance_policy_interrupted.zip \
    --timesteps 600000 \
    --lr 1e-5
```

**Console Output:**
```
📝 LOGGING ENABLED:
   CSV Log: ./logs/stage3/gated_training_20250115_143022_episodes.csv
   Summary will be saved to: ./logs/stage3/gated_training_20250115_143022_summary.json
   All logs saved to: ./logs/stage3/

🚀 GATED TRAINING STARTED
[Training proceeds with real-time logging...]
```

**What Happens:**
- ✅ Every episode logged immediately to CSV
- ✅ Data survives crashes/interruptions
- ✅ All files in `logs/stage3/` directory

---

## 📊 **STEP 2: ANALYZE LOGS & GENERATE PLOTS**

```bash
# Auto-detect latest log (recommended):
python analyze_training_logs.py --output-dir ./plots

# Or specify exact file:
python analyze_training_logs.py \
    --log logs/stage3/gated_training_20250115_143022_episodes.csv \
    --output-dir ./plots
```

**Console Output:**
```
📂 Auto-detected latest log: logs/stage3/gated_training_20250115_143022_episodes.csv
   ✅ Loaded 1003 episodes

📈 Generating learning curves...
   ✅ Saved: plots/training_learning_curves.png
   ✅ Saved: plots/training_learning_curves.pdf

📊 Generating curriculum transition plot...
   ✅ Saved: plots/curriculum_progression.png
   ✅ Saved: plots/curriculum_progression.pdf

📊 TRAINING STATISTICS SUMMARY
======================================================================
Overall Performance:
  Total Episodes:           1003
  Total Training Time:      15.2 hours
  Final Recovery Rate:      100.0% (last 50)
======================================================================

✅ Statistics saved to: plots/training_statistics.json
```

---

## 📋 **LOG FILE FORMATS:**

### **1. CSV Episode Log** 
**Location:** `logs/stage3/gated_training_TIMESTAMP_episodes.csv`

**Columns:**
```
episode, timestamp, elapsed_time_s, curriculum_level, episode_reward, 
episode_length, tumble_initiated, tumble_recovered, recovery_steps, 
disturbance_intensity, disturbance_type, rolling_10_reward, 
rolling_10_recovery_rate, rolling_50_recovery_rate
```

**Example:**
```csv
1,1705320122.5,0.8,0,12450,120,1,1,18,0.85,FLIP,12450,100.0,100.0
50,1705322890.1,2767.6,0,15230,98,1,1,15,0.92,SPIN,14820,90.0,82.0
51,1705322895.4,2772.9,1,14560,145,1,1,28,0.98,FLIP,14650,80.0,80.0
```

---

### **2. JSON Summary**
**Location:** `logs/stage3/gated_training_TIMESTAMP_summary.json`

```json
{
  "training_info": {
    "total_episodes": 1003,
    "total_time_hours": 15.2,
    "episodes_per_hour": 66.0
  },
  "final_performance": {
    "last_50_recovery_rate": 100.0,
    "avg_recovery_time_steps": 18.0,
    "avg_recovery_time_seconds": 0.9
  },
  "curriculum_progression": {
    "level_0": {
      "level_name": "EASY (0.7-0.9)",
      "reached_at_episode": 1,
      "elapsed_time_hours": 0.0
    },
    "level_1": {
      "level_name": "MEDIUM (0.9-1.1)",
      "reached_at_episode": 51,
      "elapsed_time_hours": 0.7
    },
    "level_2": {
      "level_name": "HARD (1.1-1.5)",
      "reached_at_episode": 101,
      "elapsed_time_hours": 1.5
    }
  },
  "statistics_by_level": {
    "level_0_EASY": {
      "episodes_trained": 50,
      "recovery_rate": 82.0,
      "avg_intensity": 0.85
    },
    "level_1_MEDIUM": {
      "episodes_trained": 50,
      "recovery_rate": 72.0,
      "avg_intensity": 1.02
    },
    "level_2_HARD": {
      "episodes_trained": 903,
      "recovery_rate": 100.0,
      "avg_intensity": 1.28
    }
  }
}
```

---

### **3. JSON Curriculum Progression**
**Location:** `logs/stage3/gated_training_TIMESTAMP_curriculum.json`

Contains detailed arrays for deep analysis.

---

## 🎓 **FOR YOUR PhD THESIS:**

### **Table 5.1: Complete Training Progression**
**Source:** `logs/stage3/gated_training_*_summary.json`

Extract these values:
```
Total Episodes:     1003  (training_info.total_episodes)
Training Time:      15.2h (training_info.total_time_hours)
Final Recovery:     100%  (final_performance.last_50_recovery_rate)
Avg Recovery Time:  18 steps (final_performance.avg_recovery_time_steps)
```

---

### **Table 5.2: Stage 3 Curriculum Advancement**
**Source:** `logs/stage3/gated_training_*_summary.json`

```
Level 0 (EASY):   Episode 1 → 51    (0.7h)
Level 1 (MEDIUM): Episode 51 → 101  (0.8h)
Level 2 (HARD):   Episode 101 → 1003 (13.7h)
```

---

### **Figure 5.1: Training Learning Curves**
**Source:** `plots/training_learning_curves.pdf`

4-panel plot showing:
- Episode rewards over time
- Recovery rate progression
- Curriculum level advancement
- Disturbance intensity distribution

---

### **Figure 5.2: Curriculum Progression**
**Source:** `plots/curriculum_progression.pdf`

Shows recovery rate with:
- Shaded curriculum level backgrounds
- Threshold lines (80%, 70%, 60%)
- Vertical advancement markers
- Clear performance-gated progression

---

## 💡 **TYPICAL USAGE WORKFLOW:**

```bash
# 1. Train with logging
python train_gated_curriculum_with_logging.py

# 2. Wait for training to complete (15-20 hours)

# 3. Verify logs exist
ls -la logs/stage3/
# Should show:
# - gated_training_TIMESTAMP_episodes.csv
# - gated_training_TIMESTAMP_summary.json
# - gated_training_TIMESTAMP_curriculum.json

# 4. Generate plots
python analyze_training_logs.py --output-dir ./plots

# 5. Check plots
ls -la plots/
# Should show:
# - training_learning_curves.pdf
# - curriculum_progression.pdf
# - training_statistics.json

# 6. Use in thesis!
```

---

## 🔄 **IF TRAINING IS INTERRUPTED:**

**Good News:** CSV logging is REAL-TIME!

```bash
# Check what was logged before interruption:
wc -l logs/stage3/gated_training_*_episodes.csv
# Example output: 523 lines = 522 episodes logged

# Generate plots for completed episodes:
python analyze_training_logs.py --output-dir ./plots_partial

# Resume training from last checkpoint:
python train_gated_curriculum_with_logging.py \
    --stage2-model ./models/stage3_checkpoints/gated_checkpoints/gated_curriculum_550000_steps.zip
```

---

## 📊 **VERIFY YOUR LOGS:**

```bash
# Count logged episodes:
tail -n 1 logs/stage3/gated_training_*_episodes.csv | cut -d',' -f1
# Should output: 1003 (or your total episodes)

# Check final recovery rate:
tail -n 1 logs/stage3/gated_training_*_episodes.csv | cut -d',' -f13
# Should output: 100.0 (or your final rolling 50 recovery rate)

# View summary:
cat logs/stage3/gated_training_*_summary.json | python -m json.tool | head -20

# Find curriculum advancements:
grep -E '"level_[0-9]"' logs/stage3/gated_training_*_summary.json
```

---

## ✅ **COMPLETE CHECKLIST:**

After training, you should have:

**In `logs/stage3/`:**
- [x] `gated_training_*_episodes.csv` (1000+ lines)
- [x] `gated_training_*_summary.json` (complete statistics)
- [x] `gated_training_*_curriculum.json` (detailed progression)

**In `plots/`:**
- [x] `training_learning_curves.pdf` (for thesis)
- [x] `curriculum_progression.pdf` (for thesis)
- [x] `training_statistics.json` (for analysis)

**In `models/stage3_checkpoints/curriculum_levels/`:**
- [x] `level_0_EASY_mastered.zip`
- [x] `level_1_MEDIUM_mastered.zip`

**In `models/stage3_checkpoints/`:**
- [x] `gated_curriculum_policy.zip` (final model)
- [x] `gated_curriculum_vecnormalize.pkl`

---

## 🎯 **QUICK COMMANDS REFERENCE:**

```bash
# Train
python train_gated_curriculum_with_logging.py

# Plot
python analyze_training_logs.py --output-dir ./plots

# Check logs
ls -la logs/stage3/

# View stats
cat logs/stage3/gated_training_*_summary.json | python -m json.tool

# Count episodes
wc -l logs/stage3/gated_training_*_episodes.csv

# Test model
python test_gated_curriculum.py --episodes 60
```

---

## 📥 **FILES AVAILABLE:**

1. **[train_gated_curriculum_with_logging.py](computer:///mnt/user-data/outputs/train_gated_curriculum_with_logging.py)**
   - Logs to `logs/stage3/` ✅
   
2. **[analyze_training_logs.py](computer:///mnt/user-data/outputs/analyze_training_logs.py)**
   - Reads from `logs/stage3/` ✅
   
3. **[STAGE3_LOGGING_QUICKSTART.md](computer:///mnt/user-data/outputs/STAGE3_LOGGING_QUICKSTART.md)**
   - Quick reference with correct paths ✅

---

## 🎉 **YOU'RE ALL SET!**

Just run:
```bash
python train_gated_curriculum_with_logging.py
python analyze_training_logs.py
```

All logs will be in `logs/stage3/` as requested! 🎓✨


# 🚀 QUICK START GUIDE - Stage 3 Training with Logging

## 📁 **FILE LOCATIONS:**

```
your_project/
├── train_gated_curriculum_with_logging.py    ← Use this for training
├── analyze_training_logs.py                  ← Use this for plots
│
├── logs/
│   └── stage3/                               ← ALL LOGS HERE ✅
│       ├── gated_training_TIMESTAMP_episodes.csv
│       ├── gated_training_TIMESTAMP_summary.json
│       └── gated_training_TIMESTAMP_curriculum.json
│
├── plots/                                    ← Plots saved here
│   ├── training_learning_curves.pdf
│   └── curriculum_progression.pdf
│
└── models/
    └── stage3_checkpoints/
        ├── curriculum_levels/
        │   ├── level_0_EASY_mastered.zip
        │   └── level_1_MEDIUM_mastered.zip
        └── gated_curriculum_policy.zip
```

---

## 🎯 **STEP 1: RUN TRAINING**

```bash
python train_gated_curriculum_with_logging.py \
    --stage2-model ./models/hover_disturbance_policy_interrupted.zip \
    --timesteps 600000 \
    --lr 1e-5
```

**Output:**
```
📝 LOGGING ENABLED:
   CSV Log: ./logs/stage3/gated_training_20250115_143022_episodes.csv
   Summary will be saved to: ./logs/stage3/gated_training_20250115_143022_summary.json
   All logs saved to: ./logs/stage3/

🚀 GATED TRAINING STARTED
...
```

**Logs are saved in REAL-TIME to:**
- `logs/stage3/gated_training_TIMESTAMP_episodes.csv`

**If training crashes or you interrupt, ALL DATA IS ALREADY SAVED! ✅**

---

## 📊 **STEP 2: ANALYZE & GENERATE PLOTS**

```bash
# Auto-detects latest log in logs/stage3/
python analyze_training_logs.py --output-dir ./plots

# Or specify exact file:
python analyze_training_logs.py \
    --log logs/stage3/gated_training_20250115_143022_episodes.csv \
    --output-dir ./plots
```

**Generates:**
```
plots/
├── training_learning_curves.pdf     ← For thesis Figure 5.1
├── training_learning_curves.png
├── curriculum_progression.pdf       ← For thesis Figure 5.2  
├── curriculum_progression.png
└── training_statistics.json         ← For thesis Table 5.1
```

---

## 📋 **WHAT GETS LOGGED:**

### **CSV Log (Per Episode):**
```csv
episode,timestamp,elapsed_time_s,curriculum_level,episode_reward,episode_length,
tumble_initiated,tumble_recovered,recovery_steps,disturbance_intensity,
disturbance_type,rolling_10_reward,rolling_10_recovery_rate,rolling_50_recovery_rate
```

### **JSON Summary (After Training):**
```json
{
  "training_info": {
    "total_episodes": 1003,
    "total_time_hours": 15.2,
    "episodes_per_hour": 66.0
  },
  "final_performance": {
    "last_50_recovery_rate": 100.0,
    "avg_recovery_time_steps": 18.0
  },
  "curriculum_progression": {
    "level_0": {"reached_at_episode": 1, "elapsed_time_hours": 0.0},
    "level_1": {"reached_at_episode": 51, "elapsed_time_hours": 0.7},
    "level_2": {"reached_at_episode": 101, "elapsed_time_hours": 1.5}
  },
  "statistics_by_level": {
    "level_0_EASY": {"episodes_trained": 50, "recovery_rate": 82.0},
    "level_1_MEDIUM": {"episodes_trained": 50, "recovery_rate": 72.0},
    "level_2_HARD": {"episodes_trained": 903, "recovery_rate": 100.0}
  }
}
```

---

## 🎓 **FOR YOUR THESIS:**

### **Table 5.1: Training Progression**
**Source:** `logs/stage3/gated_training_*_summary.json`

```
Total Episodes:     1003  (from training_info.total_episodes)
Training Time:      15.2h (from training_info.total_time_hours)
Final Recovery:     100%  (from final_performance.last_50_recovery_rate)
Avg Recovery Time:  18 steps (from final_performance.avg_recovery_time_steps)
```

### **Table 5.2: Curriculum Advancement**
**Source:** `logs/stage3/gated_training_*_summary.json`

```
Level 0 → 1:  Episode 51,  0.7h  (from curriculum_progression.level_1)
Level 1 → 2:  Episode 101, 1.5h  (from curriculum_progression.level_2)
```

### **Figure 5.1: Learning Curves**
**Source:** `plots/training_learning_curves.pdf`

### **Figure 5.2: Curriculum Progression**
**Source:** `plots/curriculum_progression.pdf`

---

## 🔄 **IF TRAINING CRASHES:**

**Good news:** Logs are written in REAL-TIME!

```bash
# Just run analysis on whatever was logged:
python analyze_training_logs.py --output-dir ./plots
```

You'll get plots for all episodes that completed before the crash.

---

## 📂 **FILES AFTER TRAINING:**

```bash
ls -la logs/stage3/
# Output:
# gated_training_20250115_143022_episodes.csv      ← 1003 episodes
# gated_training_20250115_143022_summary.json      ← Statistics
# gated_training_20250115_143022_curriculum.json   ← Progression

ls -la plots/
# Output:
# training_learning_curves.pdf    ← For thesis
# training_learning_curves.png
# curriculum_progression.pdf      ← For thesis
# curriculum_progression.png
# training_statistics.json
```

---

## ⚡ **QUICK COMMANDS:**

```bash
# Full training pipeline:
python train_gated_curriculum_with_logging.py && \
python analyze_training_logs.py --output-dir ./plots

# Check latest logs:
ls -lt logs/stage3/

# View summary:
cat logs/stage3/gated_training_*_summary.json | python -m json.tool

# Count episodes logged:
wc -l logs/stage3/gated_training_*_episodes.csv
```

---

## ✅ **VERIFICATION CHECKLIST:**

After training, verify you have:

- [x] `logs/stage3/gated_training_*_episodes.csv` exists
- [x] `logs/stage3/gated_training_*_summary.json` exists
- [x] `plots/training_learning_curves.pdf` exists
- [x] `plots/curriculum_progression.pdf` exists
- [x] CSV has 1000+ lines (for 1003 episodes)
- [x] Summary JSON shows your actual results

**If all checked:** You're ready for thesis! 🎓✨

---

## 🎯 **BOTTOM LINE:**

1. **Run:** `python train_gated_curriculum_with_logging.py`
2. **Analyze:** `python analyze_training_logs.py`
3. **Use plots in thesis:** `plots/*.pdf`
4. **Use stats in tables:** `logs/stage3/*_summary.json`

**All logs automatically saved to `logs/stage3/` as requested!** ✅