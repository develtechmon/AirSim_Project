"""
BEHAVIORAL CLONING TRAINING (ArduPilot pipeline, 13 observations)
=================================================================
Same job as your AirSim train_imitation.py: the network learns to copy the PID expert.

Fixes built in:
  1. Network uses SB3's layer names + ReLU, so Stage 2 PPO can load it.
  2. Input scaling saved (models/obs_norm.npz); Stage 2 uses the same numbers.
  3. Validation split by EPISODE (honest score).
  4. Transfer check: loads the weights into a real SB3 PPO and checks -> PASS.

Usage:
    python train_imitation.py                  (latest run folder, e.g. runs/run2)
    python train_imitation.py --run 1          (pick run 1)
    python train_imitation.py --run practice   (quick-test folder)
    tensorboard --logdir runs/run2/logs/stage1 (curves at http://localhost:6006)

Reads the newest dataset in runs/run<N>/demonstrations/ and saves in the SAME run folder:
    runs/run<N>/models/hover_policy_best.pth     trained network (best validation epoch)
    runs/run<N>/models/obs_norm.npz              input scaling
    runs/run<N>/models/stage1_ppo_init.zip       same network packed for Stage 2 PPO
    runs/run<N>/models/model_info.json           numbers for the thesis
    runs/run<N>/logs/stage1/training_<date>.csv  train/val loss every epoch
    runs/run<N>/logs/stage1/loss_curve_<date>.png
    runs/run<N>/logs/stage1/tb_<date>/           TensorBoard
"""

import argparse
import csv
import datetime
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OBS_DIM, ACT_DIM = 13, 3
HIDDEN = [256, 256, 128]
CLIP_OBS = 10.0
ACT_LIMIT = 5.0
RUNS = Path("runs")


def run_folder(run=None, new=False):
    """Run folders: runs/run1, runs/run2, ...  (highest number = latest).
    run given -> that folder | new=True -> next number | otherwise -> latest."""
    nums = [int(p.name[3:]) for p in RUNS.glob("run*") if p.name[3:].isdigit()]
    if run is not None:
        return RUNS / (run if run == "practice" else f"run{int(run)}")
    if new:
        return RUNS / f"run{max(nums, default=0) + 1}"
    if not nums:
        raise SystemExit("No runs/run<N> folder yet - run collect_demonstration.py first")
    return RUNS / f"run{max(nums)}"


class HoverPolicy(nn.Module):
    """13 -> 256 -> 256 -> 128 -> 3, ReLU. Layer names match SB3 PPO's actor."""

    def __init__(self):
        super().__init__()
        layers, last = [], OBS_DIM
        for h in HIDDEN:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.policy_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(last, ACT_DIM)

    def forward(self, x):
        return self.action_net(self.policy_net(x))


def normalize(x, mean, std):
    return np.clip((x - mean) / std, -CLIP_OBS, CLIP_OBS).astype(np.float32)


def transfer_check(model, val_x, save_dir):
    """Load the BC weights into a real SB3 PPO policy and prove both give the same actions."""
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("   stable-baselines3 not installed here - transfer check skipped")
        return None

    class Dummy(gym.Env):
        observation_space = spaces.Box(-CLIP_OBS, CLIP_OBS, (OBS_DIM,), np.float32)
        action_space = spaces.Box(-ACT_LIMIT, ACT_LIMIT, (ACT_DIM,), np.float32)

        def reset(self, seed=None, options=None):
            return np.zeros(OBS_DIM, np.float32), {}

        def step(self, a):
            return np.zeros(OBS_DIM, np.float32), 0.0, False, False, {}

    ppo = PPO("MlpPolicy", DummyVecEnv([Dummy]), device="cpu",
              policy_kwargs=dict(net_arch=dict(pi=HIDDEN, vf=HIDDEN), activation_fn=nn.ReLU))
    mapped = {("mlp_extractor." + k) if k.startswith("policy_net") else k: v
              for k, v in model.state_dict().items()}
    res = ppo.policy.load_state_dict(mapped, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"TRANSFER FAILED - unknown keys: {res.unexpected_keys}")
    missing = [k for k in res.missing_keys
               if k.startswith("mlp_extractor.policy_net") or k.startswith("action_net")]
    if missing:
        raise RuntimeError(f"TRANSFER FAILED - not loaded: {missing}")

    x = torch.as_tensor(val_x[:2000])
    with torch.no_grad():
        diff = float((model(x) - ppo.policy.get_distribution(x).distribution.mean).abs().max())
    ok = diff < 1e-5
    print(f"   SB3 activation: {ppo.policy.activation_fn.__name__}")
    print(f"   max |BC - PPO| action difference: {diff:.2e}  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("TRANSFER FAILED - actions differ")
    ppo.save(str(Path(save_dir) / "stage1_ppo_init"))
    return diff


def save_loss_plot(rows, best_ep, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], label="train")
    ax.plot([r[0] for r in rows], [r[2] for r in rows], label="validation")
    ax.axvline(best_ep, linestyle="--", linewidth=1, label=f"best epoch {best_ep}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Stage 1 behavioral cloning")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_folder(args.run)
    save_dir = run_dir / "models"
    log_dir = run_dir / "logs" / "stage1"

    # newest dataset of THIS run
    files = sorted((run_dir / "demonstrations").glob("expert_demonstrations_*.pkl"))
    if not files:
        raise SystemExit(f"No dataset in {run_dir / 'demonstrations'} - run collect_demonstration.py first")
    dataset = str(files[-1])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"BEHAVIORAL CLONING TRAINING (ArduPilot, 13 observations) - {run_dir.name.upper()}")
    print("=" * 70)
    print(f"Run folder: {run_dir}\n")

    print("[1/5] Loading dataset...")
    with open(dataset, "rb") as f:
        d = pickle.load(f)
    info_path = Path(dataset).with_suffix(".json")
    collect_info = json.load(open(info_path)) if info_path.exists() else {}
    X, Y, ep = d["states"], d["actions"], d["episode_ids"]
    print(f"   File: {dataset}")
    print(f"   Samples: {len(X):,} | episodes: {len(np.unique(ep))} | "
          f"sensor mode: {collect_info.get('sensor_mode', '?')}")
    assert X.shape[1] == OBS_DIM and Y.shape[1] == ACT_DIM

    print("\n[2/5] Train/validation split (by episode)...")
    eps = np.unique(ep)
    np.random.shuffle(eps)
    n_val = max(1, int(0.1 * len(eps)))
    val_eps = set(eps[:n_val].tolist())
    is_val = np.array([e in val_eps for e in ep])
    print(f"   Training: {np.sum(~is_val):,} | Validation: {np.sum(is_val):,} ({n_val} episodes)")

    mean = X[~is_val].mean(axis=0)
    std = X[~is_val].std(axis=0) + 1e-6
    np.savez(save_dir / "obs_norm.npz", mean=mean, std=std, clip=CLIP_OBS)
    Xn = normalize(X, mean, std)
    xtr, ytr = torch.as_tensor(Xn[~is_val]), torch.as_tensor(Y[~is_val])
    xva, yva = torch.as_tensor(Xn[is_val]), torch.as_tensor(Y[is_val])

    print("\n[3/5] Creating model...")
    model = HoverPolicy()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Architecture: 13 -> 256 -> 256 -> 128 -> 3 (ReLU) | {n_params:,} params")

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir / f"tb_{stamp}"))
    except Exception:
        writer = None
        print("   (TensorBoard not installed: pip install tensorboard - CSV log still saved)")

    print(f"\n[4/5] Training ({args.epochs} epochs)...")
    print("=" * 70)
    print("EPOCH | TRAIN LOSS | VAL LOSS")
    print("=" * 70)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best, best_ep, rows = float("inf"), 0, []
    csv_path = log_dir / f"training_{stamp}.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["epoch", "train_loss", "val_loss"])
    t0 = time.time()

    for e in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        tl = 0.0
        for i in range(0, len(xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(idx)
        tl /= len(xtr)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(xva), yva).item()
        rows.append((e, tl, vl))
        csv_w.writerow([e, f"{tl:.6f}", f"{vl:.6f}"])
        csv_f.flush()
        if writer:
            writer.add_scalar("loss/train", tl, e)
            writer.add_scalar("loss/val", vl, e)
        if vl < best:
            best, best_ep = vl, e
            torch.save(model.state_dict(), save_dir / "hover_policy_best.pth")
        if e == 1 or e % 10 == 0:
            print(f"{e:5d} | {tl:10.4f} | {vl:8.4f}")
    csv_f.close()
    if writer:
        writer.close()
    plot_path = log_dir / f"loss_curve_{stamp}.png"
    save_loss_plot(rows, best_ep, plot_path)
    print("=" * 70)

    model.load_state_dict(torch.load(save_dir / "hover_policy_best.pth"))
    model.eval()
    with torch.no_grad():
        per_axis = ((model(xva) - yva) ** 2).mean(dim=0).sqrt().numpy()

    print("\n[5/5] Transfer check (BC -> SB3 PPO)...")
    diff = transfer_check(model, Xn[is_val], save_dir)

    info = {
        "dataset": dataset, "collection": collect_info,
        "samples": int(len(X)), "episodes": int(len(eps)), "split": "by episode 90/10",
        "train_samples": int(np.sum(~is_val)), "val_samples": int(np.sum(is_val)),
        "architecture": "13-256-256-128-3, ReLU", "params": int(n_params),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "best_val_loss": best, "best_epoch": best_ep,
        "val_rmse_per_axis_mps": per_axis.tolist(),
        "transfer_max_action_diff": diff,
        "training_log": str(csv_path), "trained_at": stamp,
    }
    with open(save_dir / "model_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best:.4f} (epoch {best_ep})")
    print(f"Validation RMSE: vx {per_axis[0]:.3f} | vy {per_axis[1]:.3f} | vz {per_axis[2]:.3f} m/s")
    print(f"Training time: {(time.time()-t0)/60:.1f} min")
    print(f"Model:   {save_dir}/hover_policy_best.pth")
    print(f"Log:     {csv_path}")
    print(f"Plot:    {plot_path}")
    print("=" * 70)
    print(f"\nNext step: python test_hover_policy.py"
          + (" --run practice" if args.run == "practice" else f"   (uses {run_dir.name})"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="run number or 'practice' (default: latest run)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
