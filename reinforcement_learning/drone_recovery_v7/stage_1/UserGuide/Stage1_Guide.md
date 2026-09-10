# Stage 1 Guide — AirSim Behavioural Cloning

---

## Setup — Windows

- **Python:** 3.11 (required — correct gRPC communication with AirSim 1.8.1)
- **Working directory:** `drone_recovery_v7`
- **Results saved to:** `drone_recovery_v7/results/`

### Install Packages (one time only)

```
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install msgpack==0.6.2
pip install msgpack-rpc-python
pip install airsim==1.8.1
pip install numpy==1.26.4
pip install gymnasium==0.29.1
pip install stable-baselines3==2.3.2
pip install tensorboard==2.14.0 matplotlib==3.8.0 scipy==1.11.4 pandas==2.1.4
pip install sympy==1.12 pillow==10.2.0 opencv-python==4.8.1.78
pip install tqdm==4.66.1 rich==13.7.0 cloudpickle==3.0.0 typing-extensions==4.9.0
```

> ⚠️ **Never use `gymnasium[all]` or `stable-baselines3[extra]`** — they overwrite CUDA torch with CPU version.
> ⚠️ **msgpack must be 0.6.2** — newer versions break AirSim RPC communication.
> ⚠️ **numpy must be 1.26.4** — AirSim 1.8.1 breaks with numpy 2.x.

### Verify Installation

```
python -c "
import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
import numpy; print('numpy:', numpy.__version__)
import stable_baselines3; print('sb3:', stable_baselines3.__version__)
import airsim; print('airsim: ok')
print('ALL OK')
"
```

Expected output:
```
torch: 2.4.1+cu121 | CUDA: True
numpy: 1.26.4
sb3: 2.3.2
airsim: ok
ALL OK
```

---

## Launch AirSim

```
cd C:\path\to\AirSim
.\MSBuild2018.exe -ResX=640 -ResY=480 -Windowed
```

Wait until you see the drone on the field before running any script.

---

## Step 1 — Collect Expert Demonstrations at 30m

```
cd drone_recovery_v7
py .\stage_1\collect_demonstration_v2.py --episodes 200 --steps 200
```

- **Time:** ~80 minutes
- **Output:** `./demonstrations/expert_demonstrations.pkl`
- **Expected:** ~40,000 samples, reward ~5830–5870 per episode (consistent)

> ⚠️ If script hangs after "Starting collection..." use `collect_demonstration_v2_fixed.py`
> instead. Same command, same output. Fixes `client.reset()` hang on Python 3.11.

---

## Step 2 — Train BC Policy

```
cd drone_recovery_v7\stage_1
python train_imitation_v2.py | tee stage1_training_log.txt
```

- **Time:** ~2–5 minutes
- **Output:** `./models/hover_policy_best.pth`
- **Save `stage1_training_log.txt`** — contains epoch/loss numbers for Graph 1

---

## Step 3 — Collect Statistics (AirSim must be open)

```
python collect_stage1_statistics.py
```

- `collect_stage1_statistics.py` must be in same folder as `pid_expert_v2.py`
- **Time:** ~25 minutes (10 PID + 10 BC episodes × 500 steps)
- **Output:**
  - `./results/statistics_summary.pkl` — all per-step data for graphs
  - `./results/statistics_report.txt` — human-readable comparison table

---

## Step 4 — Test BC Policy

```
python test_hover_policy_v2.py --model ./models/hover_policy_best.pth --episodes 10 | tee stage1_test_results.txt
```

- **Time:** ~5 minutes
- **Save `stage1_test_results.txt`** — success rate and per-episode breakdown for paper

---

## Step 5 — Generate Graphs and Statistics

### Run these scripts after Step 3 completes:

```
python rmse_baseline.py        # Table: per-episode RMSE + transfer efficiency
python vd_distribution.py      # Graph: vd command histogram
```

---

## What Each Graph and Table Shows

### Graph 1 — Training Loss Curve
- **Data source:** `stage1_training_log.txt` (from Step 2)
- **X-axis:** Epoch (1–100)
- **Y-axis:** MSE Loss
- **Two lines:** Training loss and validation loss
- **Purpose:** Proves the BC network converged without overfitting.
  If validation loss tracks training loss and both drop consistently,
  it confirms the 90/10 split and 256-batch configuration worked correctly.
- **What to look for:** Both lines decreasing together, validation not
  diverging upward from training loss.

---

### Graph 2 — Mean Episode Altitude (PID vs BC)
- **Data source:** `statistics_summary.pkl` → `episode_altitudes`
- **X-axis:** Timestep (0–500 steps = 25 seconds)
- **Y-axis:** Altitude in metres
- **Two lines:** PID Expert (baseline) vs BC Policy
- **Purpose:** Shows whether the BC policy tracks the expert's altitude
  behaviour in closed-loop AirSim control. If the BC line stays close
  to the PID line near 30m, the imitation was successful.
- **What to look for:** BC line hovering near 30m with similar stability
  profile to the PID expert.

---

### Graph 3 — RMSE Per Episode (PID vs BC)
- **Data source:** `statistics_summary.pkl` → `ep_rmse`
- **X-axis:** Episode number (1–10)
- **Y-axis:** RMSE in metres
- **Two grouped bars per episode:** PID Expert and BC Policy
- **Purpose:** Proves BC performance is consistent across all 10 episodes,
  not just lucky on one or two. Low and consistent RMSE across all episodes
  means the policy is reliable, not occasionally good.
- **What to look for:** BC bars close to PID bars across all episodes,
  with low variance between episodes.

---

### Graph 4 — Maximum Horizontal Drift Per Episode (PID vs BC)
- **Data source:** `statistics_summary.pkl` → `ep_dist_max`
- **X-axis:** Episode number (1–10)
- **Y-axis:** Max drift from (0,0) position in metres
- **Two grouped bars per episode:** PID Expert and BC Policy
- **Purpose:** Proves the BC policy holds horizontal position, not just
  altitude. A policy that holds altitude but drifts horizontally is only
  half-trained. This graph exposes that failure if it exists.
- **What to look for:** BC bars similar to or lower than PID bars.
  Horizontal drift much larger than PID would indicate the policy did not
  learn position-hold, only altitude-hold.

---

### Graph 5 — Vertical Command Distribution (vd histogram)
- **Data source:** `statistics_summary.pkl` → `all_actions[:, 2]`
- **X-axis:** vd command value in m/s
- **Y-axis:** Frequency (count of steps with that command value)
- **Purpose:** Proves BC learned the same vertical control strategy as the
  PID expert, not just a different mechanism that happens to achieve similar
  altitude. If the BC histogram shape, peak location and spread are similar
  to the PID expert, the imitation was faithful at the control action level.
- **What to look for:** BC histogram roughly matching PID histogram shape.
  A BC histogram centred near zero with narrow spread means the policy makes
  small, precise corrections — the same behaviour as the expert.

---

### Table — RMSE Comparison and Transfer Efficiency
- **Data source:** `statistics_summary.pkl` → `comparison`
- **Script:** `rmse_baseline.py`
- **Purpose:** Single most important summary number for the paper.
  Transfer efficiency = how much of the PID expert's performance was
  preserved in the BC policy. Formula:
  `transfer_efficiency = (1 - (BC_RMSE - PID_RMSE) / PID_RMSE) × 100%`
- **Target:** ≥ 90% transfer efficiency confirms BC successfully
  learned from the expert.
- **Goes in paper:** Results Table comparing Stage 1 BC performance
  against PID baseline.

---

## Output Files Checklist

| File | Used for |
|---|---|
| `expert_demonstrations.pkl` | Training data for BC |
| `hover_policy_best.pth` | Trained BC model |
| `stage1_training_log.txt` | Graph 1 — loss curve |
| `statistics_summary.pkl` | Graphs 2–5 and RMSE table |
| `statistics_report.txt` | Quick reference table |
| `stage1_test_results.txt` | Success rate for paper |

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| Script hangs at "Starting collection..." | `client.reset()` deadlock on Python 3.11 | Use `collect_demonstration_v2_fixed.py` |
| `ModuleNotFoundError: msgpackrpc` | msgpack-rpc-python not installed | `pip install msgpack==0.6.2 msgpack-rpc-python` |
| `TypeError: encoding keyword` | msgpack version too new | `pip install msgpack==0.6.2` |
| torch CUDA not available | Wrong torch version installed | Reinstall with `--index-url https://download.pytorch.org/whl/cu121` |
| AirSim "API call not received" | Drone spawned inside collision mesh | Move PlayerStart actor up in Unreal |
| gymnasium overwrites torch | Used `gymnasium[all]` flag | `pip install gymnasium==0.29.1` (no [all]) |
