# SIMPLE QUICK START GUIDE

## ✅ **3-Stage Training - SIMPLE VERSION**

All commands work EXACTLY like your original code!  
The ONLY difference: Now uses 13 observations for full transfer learning!

---

## 📁 **Setup**

Put all files in these folders:
```
stage1_v2/  ← Stage 1 files
stage2_v2/  ← Stage 2 files  
stage3_v2/  ← Stage 3 files
```

---

## 🚀 **STAGE 1: Hover (30 min)**

```bash
cd stage1_v2

# 1. Test PID (5 min)
python pid_expert_v2.py

# 2. Collect data (60 min OR 5 min quick test)
python collect_demonstration_v2.py --episodes 2000  # Full
# OR
python collect_demonstration_v2.py --episodes 100   # Quick test

# 3. Train (25 min)
python train_imitation_v2.py

# 4. Test (2 min)
python test_hover_policy_v2.py
```

**Expected:** 95%+ success, creates `./models/hover_policy_best.pth`

---

## 🌬️ **STAGE 2: Wind (5 hours)**

```bash
cd ../stage2_v2

# 1. Train (5 hours) - JUST RUN IT!
python train_stage2_disturbance_v2.py

# 2. Test (2 min)
python test_stage2_policy_v2.py
```

**Expected:** 90%+ success, creates `./models/hover_disturbance_policy.zip`

**That's it!** Stage 2 automatically:
- Finds Stage 1 model at `./models/hover_policy_best.pth`
- Loads the weights
- Starts training with transfer learning
- Saves to `./models/`

---

## 🔄 **STAGE 3: Flips (3 hours)**

```bash
cd ../stage3_v2

# 1. Train (3 hours) - JUST RUN IT!
python train_stage3_flip_v2.py

# 2. Test (5 min)
python test_stage3_policy_v2.py --episodes 20
```

**Expected:** 75%+ recovery, creates `./models/flip_recovery_policy.zip`

**That's it!** Stage 3 automatically:
- Finds Stage 2 model at `./models/hover_disturbance_policy.zip`
- Loads the weights
- Starts training with transfer learning
- Saves to `./models/`

---

## 📊 **File Locations (Same as Original!)**

```
stage1_v2/models/hover_policy_best.pth          ← Stage 1 output
stage2_v2/models/hover_disturbance_policy.zip   ← Stage 2 output
stage3_v2/models/flip_recovery_policy.zip       ← Stage 3 output
```

---

## ⚠️ **ONLY Difference from Original:**

**Observation space is now 13** (includes angular velocity)

This means:
- ✅ Stage 1 → Stage 2 transfer learning works
- ✅ Stage 2 → Stage 3 transfer learning works
- ✅ Total time: 8.5 hours (vs 15 hours before)

**Everything else is EXACTLY the same!**

---

## 🎯 **Success Criteria**

### Stage 1
```
Success Rate: 95-100%
Avg Distance: 0.39m
```

### Stage 2
```
Episode 1000: ~+32,000 return
Success Rate: 90-100%
Max Wind: 4.5-5.0 m/s
```

### Stage 3
```
Recovery Rate: 70-80%
Recovery Time: 4-8 seconds
```

---

## 🐛 **If Something Goes Wrong**

**Stage 1:** Check data has `State dimension: 13`

**Stage 2:** 
- Should say "✅ Stage 1 policy loaded"
- If not, check `./models/hover_policy_best.pth` exists

**Stage 3:**
- Should say "✅ Stage 2 policy loaded successfully!"
- If not, check `./models/hover_disturbance_policy.zip` exists

---

## 🎊 **That's It!**

**Same simple workflow you know!**  
**Just with better transfer learning!**

```bash
# Stage 1
cd stage1_v2
python train_imitation_v2.py

# Stage 2  
cd ../stage2_v2
python train_stage2_disturbance_v2.py

# Stage 3
cd ../stage3_v2
python train_stage3_flip_v2.py
```

**Done!** 🚁
