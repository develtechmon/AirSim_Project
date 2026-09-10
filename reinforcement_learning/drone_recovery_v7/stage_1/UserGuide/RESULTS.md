## Results 

pid_expert.py

command used : `python pid_expert.py`

### Result
```
======================================================================
🧪 TESTING PID EXPERT CONTROLLER (13 OBSERVATIONS)
======================================================================

Connected!
Client Ver:1 (Min Req: 1), Server Ver:1 (Min Req: 1)

Taking off...
Moving to 30m altitude...
✓ PID Expert Controller Initialized (13 observations)
  Target: Hover at 30.0m
  Control frequency: 20 Hz
  Observation space: 13 (includes angular velocity)

🎯 Running PID hover test for 100 steps (5 seconds)...
Watch the drone - it should hover stably!

Step   0: Alt=30.38m, Dist from center=0.00m
Step  20: Alt=30.27m, Dist from center=0.00m
Step  40: Alt=30.22m, Dist from center=0.00m
Step  60: Alt=30.17m, Dist from center=0.00m
Step  80: Alt=30.14m, Dist from center=0.00m

======================================================================
📊 RESULTS
======================================================================
Mean altitude: 30.198m (target: 30.0m)
Std deviation: 0.062m
Max error: 0.384m

✅ PID Expert is EXCELLENT! Ready to generate demonstrations.
======================================================================

```

collect_demonstration.py

command used : `python collect_demonstration_v2.py --episodes 200 --steps 200 (i'm using this command togenerate 4000 samples)`

### Result
```
======================================================================
📊 COLLECTING EXPERT DEMONSTRATIONS
======================================================================
Target episodes: 200
Steps per episode: 200
Total data points: ~40,000
Save directory: ./demonstrations
======================================================================

Connected!
Client Ver:1 (Min Req: 1), Server Ver:1 (Min Req: 1)

✓ PID Expert Controller Initialized (13 observations)
  Target: Hover at 30.0m
  Control frequency: 20 Hz
  Observation space: 13 (includes angular velocity)
Starting collection...

Episode   10/200 | Avg Reward:  5834.8 | Speed: 2.5 eps/min | ETA: 75 min
Episode   20/200 | Avg Reward:  5831.7 | Speed: 2.5 eps/min | ETA: 71 min
Episode   30/200 | Avg Reward:  5850.8 | Speed: 2.5 eps/min | ETA: 67 min
Episode   40/200 | Avg Reward:  5838.5 | Speed: 2.5 eps/min | ETA: 63 min
Episode   50/200 | Avg Reward:  5836.4 | Speed: 2.5 eps/min | ETA: 59 min
Episode   60/200 | Avg Reward:  5828.3 | Speed: 2.5 eps/min | ETA: 55 min
Episode   70/200 | Avg Reward:  5851.4 | Speed: 2.5 eps/min | ETA: 51 min
Episode   80/200 | Avg Reward:  5869.0 | Speed: 2.5 eps/min | ETA: 47 min
Episode   90/200 | Avg Reward:  5828.0 | Speed: 2.5 eps/min | ETA: 43 min
Episode  100/200 | Avg Reward:  5832.9 | Speed: 2.5 eps/min | ETA: 39 min
Episode  110/200 | Avg Reward:  5843.3 | Speed: 2.5 eps/min | ETA: 36 min
Episode  120/200 | Avg Reward:  5833.0 | Speed: 2.5 eps/min | ETA: 32 min
Episode  130/200 | Avg Reward:  5855.7 | Speed: 2.5 eps/min | ETA: 28 min
Episode  140/200 | Avg Reward:  5828.0 | Speed: 2.5 eps/min | ETA: 24 min
Episode  150/200 | Avg Reward:  5823.7 | Speed: 2.5 eps/min | ETA: 20 min
Episode  160/200 | Avg Reward:  5836.7 | Speed: 2.5 eps/min | ETA: 16 min
Episode  170/200 | Avg Reward:  5824.2 | Speed: 2.5 eps/min | ETA: 12 min
Episode  180/200 | Avg Reward:  5838.2 | Speed: 2.5 eps/min | ETA: 8 min
Episode  190/200 | Avg Reward:  5856.7 | Speed: 2.5 eps/min | ETA: 4 min
Episode  200/200 | Avg Reward:  5846.6 | Speed: 2.5 eps/min | ETA: 0 min

======================================================================
💾 SAVING FINAL DATASET
======================================================================

📊 Dataset Statistics:
   Total samples: 40,000
   State dimension: 13
   Action dimension: 3
   Mean episode reward: 5839.4
   Std episode reward: 33.5
   Collection time: 79.1 minutes

💾 Saved to: ./demonstrations/expert_demonstrations.pkl
   File size: 4.6 MB
======================================================================


✅ Collection complete!
Next step: Run train_imitation_v2.py to train the policy

```

train_imitation.py

command used : `python train_imitation.py`

### Result

```
======================================================================
🎓 BEHAVIORAL CLONING TRAINING (13 OBSERVATIONS)
======================================================================

Using device: cuda

[1/5] Loading dataset...
   Total samples: 40,000
   State dimension: 13 ← Should be 13!
   Action dimension: 3
   Mean episode reward: 5839.4
   ✅ Observation space verified: 13 dimensions

[2/5] Creating train/validation split...
   Training samples: 36,000
   Validation samples: 4,000

[3/5] Creating model...
   Model parameters: 102,659
   Architecture: 13 → 256 → 256 → 128 → 3
   ✅ Compatible with Stage 2 & 3 (same architecture)

[4/5] Training...
   Epochs: 100
   Batch size: 256
   Learning rate: 0.001

======================================================================
EPOCH | TRAIN LOSS | VAL LOSS | TIME
======================================================================
    1 |     0.4493 |   0.4515 |  1.4s
   10 |     0.1318 |   0.1461 |  0.7s
   20 |     0.0687 |   0.0780 |  0.7s
   30 |     0.0417 |   0.0427 |  0.9s
   40 |     0.0314 |   0.0353 |  0.8s
   50 |     0.0261 |   0.0273 |  0.7s
   60 |     0.0228 |   0.0222 |  0.9s
   70 |     0.0192 |   0.0215 |  0.8s
   80 |     0.0173 |   0.0207 |  0.8s
   90 |     0.0153 |   0.0164 |  0.7s
  100 |     0.0145 |   0.0158 |  0.7s
======================================================================

[5/5] Saving model...
   ✅ Best model: ./models/hover_policy_best.pth
   ✅ Final model: ./models/hover_policy_final.pth
   ✅ Model info: ./models/model_info.pkl

======================================================================
📊 TRAINING COMPLETE
======================================================================
Best Validation Loss: 0.0146
Final Training Loss: 0.0145
Total Training Time: 1.3 minutes

📈 Estimated Success Rate: 95%+

✅ Next step: Run test_hover_policy_v2.py to evaluate!
======================================================================

```

test_hover_policy.py

command used : `python test_hover_policy.py`

### Result
```
   ✅ Model loaded successfully
   📊 Observation space: 13 (pos + vel + ori + ang_vel)

[2/3] Connecting to AirSim...
Connected!
Client Ver:1 (Min Req: 1), Server Ver:1 (Min Req: 1)

   ✅ Connected

[3/3] Running 10 test episodes...
   Max steps per episode: 500
   Target altitude: 30.0m

======================================================================
Episode  1/10 | Steps: 500 | Success: ✅ | Avg Dist: 0.06m | Reason: completed
Episode  2/10 | Steps: 500 | Success: ✅ | Avg Dist: 0.04m | Reason: completed
Episode  3/10 | Steps: 500 | Success: ✅ | Avg Dist: 0.05m | Reason: completed
Episode  4/10 | Steps: 500 | Success: ✅ | Avg Dist: 0.05m | Reason: completed
Episode  5/10 | Steps: 500 | Success: ✅ | Avg Dist: 0.06m | Reason: completed
Episode  6/10 | Steps: 500 | Success: ✅ | Avg Dist: 0.07m | Reason: completed
Episode  7/10 | Steps: 500 | Success: ✅ | Avg Dist: 0.06m | Reason: completed
Episode  8/10 | Steps: 500 | Success: ✅ | Avg Dist: 0.05m | Reason: completed
Episode  9/10 | Steps: 500 | Success: ✅ | Avg Dist: 0.06m | Reason: completed
Episode 10/10 | Steps: 500 | Success: ✅ | Avg Dist: 0.06m | Reason: completed
======================================================================

======================================================================
📊 TEST RESULTS
======================================================================
Success Rate: 100% (10/10 episodes)
Average Episode Length: 500.0 steps
Average Distance from Target: 0.06m (successful episodes)

======================================================================
✅ EXCELLENT! Policy successfully learned to hover!
   Ready for Stage 2 (disturbance recovery)
   ✅ Model can be used for transfer learning!
======================================================================
```