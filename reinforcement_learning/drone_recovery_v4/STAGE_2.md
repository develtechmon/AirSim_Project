# 🌬️ STAGE 2: DISTURBANCE RECOVERY - COMPLETE GUIDE

## ✅ **WHAT YOU'VE ACCOMPLISHED SO FAR**

```
[✅] Stage 1 Complete!
     - PID Expert: EXCELLENT
     - Data Collection: 20,000 samples
     - Training: Val loss 0.0380
     - Testing: 100% success rate!
     
[🚀] Stage 2: Starting Now!
     - Add wind disturbances
     - Train with PPO
     - Expected: 80%+ success with wind
```

---

## 🎯 **STAGE 2 OVERVIEW**

### **Goal:**
Train your drone to maintain stable hover despite random wind gusts

### **What's New:**
- **Wind System:** Random gusts from 0-5 m/s
- **Wind Changes:** Every 1-3 seconds
- **Challenge:** Drone must actively compensate

### **Training Method:**
- Starts from your 100% hover policy (Stage 1)
- Uses PPO (Reinforcement Learning)
- Fine-tunes to handle disturbances
- Training time: 2-3 hours

---

## 📋 **STEP-BY-STEP INSTRUCTIONS**

### **Step 1: Test the Disturbance Environment (Optional - 5 min)**

**Command:**
```bash
python drone_hover_disturbance_env.py
```

**What This Does:**
- Tests the wind system
- Shows how environment works
- Runs 50 random actions

**Expected Output:**
```
Testing disturbance environment...
✓ Drone Hover Disturbance Environment
  - Wind strength: 0-5.0 m/s

🔄 RESET
   Initial wind: [2.3, -1.5, 0.4] m/s

[Step 50] Alt=9.2m | Dist=1.34m | Wind=3.2m/s | Reward=-12.3
   🌬️  Wind changed: [2.3, -1.5, 0.4] → [4.1, 0.8, -0.6]

✅ Environment test complete!
```

**This is optional - you can skip to Step 2 if you want!**

---

### **Step 2: Start PPO Training (2-3 hours)**

**Command:**
```bash
python train_stage2_disturbance.py
```

**What This Does:**
1. Loads your Stage 1 hover policy (100% success)
2. Transfers weights to PPO actor network
3. Adds wind to environment
4. Trains with PPO for 500,000 timesteps

**Expected Output:**
```
======================================================================
🌬️  STAGE 2: DISTURBANCE RECOVERY TRAINING
======================================================================
Training drone to handle wind while hovering
Starting from Stage 1 policy (100% success)
Expected training time: 2-3 hours
======================================================================

[1/5] Loading Stage 1 policy: ./models/hover_policy_best.pth
   ✅ Stage 1 policy loaded
   📊 This policy achieved 100% hover success!

[2/5] Creating PPO model...
   ✅ PPO model created

[3/5] Loading pretrained weights into PPO actor...
   ✅ Pretrained weights loaded into actor network
   💡 PPO will start from 100% hover success!
   📈 Only needs to learn wind compensation

[4/5] Creating disturbance environment...
   Wind strength: 0-5.0 m/s
   ✅ Environment created with wind disturbances

[5/5] Starting PPO training...
   Total timesteps: 500,000
   Learning rate: 3e-05
   Estimated time: 16.7 hours (at ~30k steps/hour)

======================================================================
🚀 TRAINING STARTED
======================================================================
Watch for episode statistics every 10 episodes...
Model will learn to compensate for wind disturbances!
======================================================================

======================================================================
📊 EPISODE 10
======================================================================
   Last 10 Episodes:
      Avg Return: 1234.5
      Avg Length: 287.3 steps
      Max Length: 456 steps
   Current wind: 3.2 m/s
======================================================================

[Progress continues...]
```

**Training Progress to Watch:**

| Episodes | Expected Behavior |
|----------|------------------|
| 1-50 | Learning wind compensation, some failures |
| 51-150 | Improving stability, 50-70% success |
| 151-300 | Mastering compensation, 70-85% success |
| 300+ | Robust to wind, 80%+ success |

---

### **Step 3: Test the Policy (5-10 min)**

**After training completes, run:**

```bash
python test_stage2_policy.py
```

**Expected Output:**
```
🧪 TESTING STAGE 2: DISTURBANCE RECOVERY
======================================================================

[1/3] Loading model: ./models/hover_disturbance_policy.zip
   ✅ Model loaded

[2/3] Running 10 test episodes...
   Wind strength: 0-5.0 m/s
   Max steps: 500 per episode

======================================================================
Episode  1/10 | Steps: 500 | Success: ✅ | Dist: 0.52m | Wind: 2.8m/s (max: 4.2) | Reason: completed
Episode  2/10 | Steps: 500 | Success: ✅ | Dist: 0.48m | Wind: 3.1m/s (max: 4.7) | Reason: completed
Episode  3/10 | Steps: 500 | Success: ✅ | Dist: 0.61m | Wind: 2.5m/s (max: 4.1) | Reason: completed
Episode  4/10 | Steps: 387 | Success: ❌ | Dist: 1.89m | Wind: 3.8m/s (max: 5.0) | Reason: out_of_bounds
Episode  5/10 | Steps: 500 | Success: ✅ | Dist: 0.55m | Wind: 2.9m/s (max: 4.5) | Reason: completed
Episode  6/10 | Steps: 500 | Success: ✅ | Dist: 0.49m | Wind: 3.2m/s (max: 4.3) | Reason: completed
Episode  7/10 | Steps: 500 | Success: ✅ | Dist: 0.58m | Wind: 2.7m/s (max: 4.6) | Reason: completed
Episode  8/10 | Steps: 500 | Success: ✅ | Dist: 0.47m | Wind: 3.0m/s (max: 4.4) | Reason: completed
Episode  9/10 | Steps: 500 | Success: ✅ | Dist: 0.53m | Wind: 3.3m/s (max: 4.8) | Reason: completed
Episode 10/10 | Steps: 500 | Success: ✅ | Dist: 0.51m | Wind: 2.6m/s (max: 4.2) | Reason: completed
======================================================================

📊 TEST RESULTS
======================================================================
Success Rate: 90% (9/10 episodes)
Average Distance: 0.53m (successful episodes)
Average Wind Handled: 2.9 m/s
Maximum Wind Survived: 5.0 m/s
Average Episode Length: 488.7 steps

✅ EXCELLENT! Policy handles wind disturbances very well!
   Ready for Stage 3 (flip recovery)
======================================================================

📊 COMPARISON TO STAGE 1
======================================================================
Stage 1 (no wind):  100% success, 0.39m avg distance
Stage 2 (with wind): 90% success, 0.53m avg distance

✅ Successfully maintained hover ability despite wind!
======================================================================
```

---

## 🎯 **SUCCESS CRITERIA**

### **Excellent ✅ (Ready for Stage 3)**
- Success rate ≥ 80%
- Avg distance < 0.7m
- Can handle wind up to 4+ m/s

### **Good ⚠️ (Can proceed or train longer)**
- Success rate 70-79%
- Avg distance 0.7-1.0m
- Struggles with high wind (>4 m/s)

### **Needs Work ❌ (Train longer)**
- Success rate < 70%
- Avg distance > 1.0m
- Frequently goes out of bounds

---

## ⏰ **TRAINING TIME ESTIMATES**

```
500,000 timesteps @ 30,000 steps/hour = ~17 hours

BUT with progress bar you can stop early if:
- Episode 100: Already 80%+ success → Stop and test!
- Episode 200: Plateauing at 85%+ → Good enough!
- Episode 300: Not improving → May need adjustments
```

**You can stop training early if performance looks good!**

To stop: `Ctrl+C` (model will be saved)

---

## 🔧 **TRAINING OPTIONS**

### **Standard Training (Recommended):**
```bash
python train_stage2_disturbance.py
```

### **Shorter Training (Quick test):**
```bash
python train_stage2_disturbance.py --timesteps 250000
```
(~8 hours, may achieve 70-80% success)

### **Longer Training (If struggling):**
```bash
python train_stage2_disturbance.py --timesteps 750000
```
(~25 hours, for 85%+ success)

### **Easier Wind (If too hard):**
```bash
python train_stage2_disturbance.py --wind-strength 3.0
```
(Reduces wind to 0-3 m/s instead of 0-5 m/s)

### **Custom Learning Rate:**
```bash
python train_stage2_disturbance.py --lr 5e-5
```
(Higher LR = faster learning but less stable)

---

## 📂 **FILES CREATED**

```
models/
├── hover_disturbance_policy.zip              ← Final Stage 2 model
├── hover_disturbance_vecnormalize.pkl        ← Normalization stats
└── stage2_checkpoints/
    ├── disturbance_policy_25000_steps.zip    ← Checkpoint 1
    ├── disturbance_policy_50000_steps.zip    ← Checkpoint 2
    └── ...

logs/
└── stage2/
    └── PPO_*/
        ├── events.out.tfevents.*              ← Tensorboard logs
        └── ...
```

---

## 📊 **MONITORING TRAINING**

### **In Terminal:**
Watch for episode statistics every 10 episodes:
```
📊 EPISODE 100
   Last 10 Episodes:
      Avg Return: 2134.5    ← Should increase over time
      Avg Length: 387.3     ← Should increase over time
```

### **With Tensorboard:**
```bash
tensorboard --logdir=./logs/stage2/
```
Open: http://localhost:6006

**Graphs to watch:**
- `rollout/ep_rew_mean` → Should increase
- `rollout/ep_len_mean` → Should increase
- `train/policy_loss` → Should stabilize

---

## 🚨 **TROUBLESHOOTING**

### **Problem: Training very slow**
**Solution:** Training on CPU is slow. This is expected.
- Can run overnight
- Or reduce timesteps: `--timesteps 250000`

### **Problem: Can't load Stage 1 weights**
**Solution:** Check file exists:
```bash
dir models\hover_policy_best.pth
```
If missing, retrain Stage 1

### **Problem: Success rate not improving**
**Solutions:**
1. Train longer: `--timesteps 750000`
2. Reduce wind: `--wind-strength 3.0`
3. Lower LR: `--lr 1e-5`

### **Problem: Out of memory**
**Solution:** Close other programs, restart AirSim

---

## 💬 **WHAT TO SHARE WITH ME**

### **After Training:**
```
Final episode statistics:
   Avg Return: ???
   Avg Length: ???

Test results:
   Success Rate: ??%
   Average Distance: ?.??m
   Max Wind Survived: ?.?m/s
```

---

## 🎯 **AFTER STAGE 2**

### **If Success ≥ 80%:**
**You're ready for Stage 3 (Flip Recovery)!**

I'll create:
- Flip recovery environment
- Curriculum training script (30° → 180°)
- Testing scripts

### **If Success 70-79%:**
**You can either:**
- Continue to Stage 3 (might be harder)
- Train Stage 2 longer
- Test with easier wind first

### **If Success < 70%:**
**Recommended:**
- Train longer (750k timesteps)
- Or reduce wind strength
- Then retest before Stage 3

---

## 📋 **QUICK COMMAND REFERENCE**

```bash
# Test environment (optional)
python drone_hover_disturbance_env.py

# Train Stage 2 (standard)
python train_stage2_disturbance.py

# Train Stage 2 (quick)
python train_stage2_disturbance.py --timesteps 250000

# Test trained policy
python test_stage2_policy.py

# Monitor training
tensorboard --logdir=./logs/stage2/
```

---

## 🎓 **KEY CONCEPTS**

### **Transfer Learning:**
We don't train from scratch! We start with your 100% hover policy and only learn wind compensation.

### **PPO (Proximal Policy Optimization):**
- More sample-efficient than random RL
- Stable updates
- Good for fine-tuning

### **Why This Works:**
Stage 1 solved hover → Stage 2 only solves wind  
(Much easier than solving both at once!)

---

**Start training now! Run: `python train_stage2_disturbance.py`** 🌬️🚁✨

**This will take 2-3 hours. You can let it run overnight!**