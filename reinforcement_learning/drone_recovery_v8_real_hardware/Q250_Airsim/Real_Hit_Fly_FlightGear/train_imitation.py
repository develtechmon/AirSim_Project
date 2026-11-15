r"""
BEHAVIORAL CLONING - RECOVERY  (Stage 3 step 3b, copies the attitude-rule teacher)
=================================================================================

1. HOW TO RUN  (PowerShell in Stage_3 - no SITL / AirSim needed, ~5-10 min on the RTX 3080)
------------------------------------------------------------------------
    Train:

    py .\train_imitation.py

    Watch the curves while it trains (optional, second PowerShell, then open http://localhost:6006):

    tensorboard --logdir logs\bc


2. EXPECTED RESULT
------------------------------------------------------------------------
    Dataset (collect 2026-10-19): 400 hits, 400 KEPT, 37,311 samples
    Printed every 10 epochs:   EPOCH | TRAIN LOSS | VAL LOSS   (both go down, then flatten)
    At the end (targets - first run will give the real numbers):
        Validation RMSE   roll < 0.05 rad (~3 deg) | pitch < 0.05 rad | thrust < 0.10 x hover
        Transfer check    max |BC - PPO| < 1e-5  -> PASS

    How to read it / what to do:
    Result                                   Meaning                         Action
    RMSE under targets, Transfer PASS        network copies the teacher      next: BC check in SITL
    VAL LOSS goes UP while TRAIN goes down   memorising, not learning        send me the CSV log
    RMSE thrust > 0.2 x hover                thrust not learned              send me the CSV log
    Transfer FAIL                            PPO cannot load the network     send me the printout


3. OUTPUT FILES
------------------------------------------------------------------------
    models\recovery_policy_best.pth      trained network (best validation epoch)
    models\obs_norm.npz                  input scaling (mean / std) - PPO, test and Pi use the SAME
    models\stage3_ppo_init.zip           the same network packed for Stage 3 PPO
    models\model_info.json               all numbers for the thesis
    logs\bc\training_<date>.csv          train / validation loss every epoch
    logs\bc\loss_curve_<date>.png        loss curve (for the thesis)
    logs\bc\tb_<date>\                   TensorBoard


4. WHAT THE SCRIPT DOES  (same as Stage 1 train_imitation.py, new data)
------------------------------------------------------------------------
    Step 1  load demos\recovery_demos.npz (the "photo album" from collect_demonstration.py)
    Step 2  split by HIT: 90 % of hits to learn from, 10 % hidden for an honest test
    Step 3  scale the 13 inputs (mean / std) and save that scaling
    Step 4  train the network: "for these 13 numbers, give the same 3 numbers as the teacher"
    Step 5  check the error per action and per impact type (flip / bird / spin)
    Step 6  transfer check: load the weights into a real SB3 PPO -> must give identical actions


5. NETWORK
------------------------------------------------------------------------
    13 inputs -> 256 -> 256 -> 128 -> 3 outputs, ReLU   (same size as Stage 1)
    Layer names match SB3 PPO's actor, so Stage 3 PPO starts from this network.


6. INPUTS AND OUTPUTS  (same as collect_demonstration.py section 6)
------------------------------------------------------------------------
    IN  (13)  down_x down_y down_z | p q r | v_down | dalt_since_hit | v_fwd v_right |
              prev_roll prev_pitch prev_thr_x_hover
    OUT (3)   roll_rad | pitch_rad | thrust_x_hover


7. SETTINGS  (--option, default)
------------------------------------------------------------------------
    --epochs 150   --batch-size 256   --lr 0.001   --seed 0   --demos demos\recovery_demos.npz
"""

import argparse
import csv
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OBS_DIM, ACT_DIM = 13, 3
HIDDEN = [256, 256, 128]
CLIP_OBS = 10.0
ACT_LIMIT = 5.0
ACT_NAMES = ["roll_rad", "pitch_rad", "thrust_x_hover"]
MODELS, LOGS = Path("models"), Path("logs") / "bc"


class RecoveryPolicy(nn.Module):
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
    print(f"   max |BC - PPO| action difference: {diff:.2e}  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError("TRANSFER FAILED - actions differ")
    ppo.save(str(Path(save_dir) / "stage3_ppo_init"))
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
    ax.set_title("Stage 3 behavioral cloning (recovery, teacher = attitude-rule)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hit_types(demos_path):
    """hit number -> impact type, from collect_episodes.csv next to the npz (for per-type errors)."""
    f = Path(demos_path).with_name("collect_episodes.csv")
    if not f.exists():
        return {}
    with open(f) as fh:
        return {int(r["hit"]): r["type"] for r in csv.DictReader(fh)}


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not Path(args.demos).exists():
        raise SystemExit(f"No {args.demos} - run collect_demonstration.py first")
    MODELS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("BEHAVIORAL CLONING - RECOVERY (Stage 3, teacher = attitude-rule)")
    print("=" * 70)

    print("[1/6] Loading dataset...")
    d = np.load(args.demos)
    X, Y, ep = d["obs"].astype(np.float32), d["act"].astype(np.float32), d["episode"]
    info_path = Path(args.demos).with_name("collect_info.json")
    collect_info = json.load(open(info_path)) if info_path.exists() else {}
    kinds = hit_types(args.demos)
    print(f"   File: {args.demos}")
    print(f"   Samples: {len(X):,} | hits: {len(np.unique(ep))} | "
          f"hover thrust: {collect_info.get('hover_thrust', '?')}")
    assert X.shape[1] == OBS_DIM and Y.shape[1] == ACT_DIM

    print("\n[2/6] Train/validation split (by hit)...")
    eps = np.unique(ep)
    np.random.shuffle(eps)
    n_val = max(1, int(0.1 * len(eps)))
    val_eps = set(eps[:n_val].tolist())
    is_val = np.array([e in val_eps for e in ep])
    print(f"   Training: {np.sum(~is_val):,} | Validation: {np.sum(is_val):,} ({n_val} hits)")

    print("\n[3/6] Input scaling...")
    mean = X[~is_val].mean(axis=0)
    std = X[~is_val].std(axis=0) + 1e-6
    np.savez(MODELS / "obs_norm.npz", mean=mean, std=std, clip=CLIP_OBS)
    Xn = normalize(X, mean, std)
    xtr, ytr = torch.as_tensor(Xn[~is_val]), torch.as_tensor(Y[~is_val])
    xva, yva = torch.as_tensor(Xn[is_val]), torch.as_tensor(Y[is_val])
    print(f"   Saved {MODELS / 'obs_norm.npz'}")

    model = RecoveryPolicy()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Network: 13 -> 256 -> 256 -> 128 -> 3 (ReLU) | {n_params:,} params")
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(LOGS / f"tb_{stamp}"))
    except Exception:
        writer = None
        print("   (TensorBoard not installed: pip install tensorboard - CSV log still saved)")

    print(f"\n[4/6] Training ({args.epochs} epochs)...")
    print("=" * 70)
    print("EPOCH | TRAIN LOSS | VAL LOSS")
    print("=" * 70)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best, best_ep, rows = float("inf"), 0, []
    csv_path = LOGS / f"training_{stamp}.csv"
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
            torch.save(model.state_dict(), MODELS / "recovery_policy_best.pth")
        if e == 1 or e % 10 == 0:
            print(f"{e:5d} | {tl:10.5f} | {vl:8.5f}")
    csv_f.close()
    if writer:
        writer.close()
    plot_path = LOGS / f"loss_curve_{stamp}.png"
    save_loss_plot(rows, best_ep, plot_path)
    print("=" * 70)

    print("\n[5/6] Error per action and per impact type (validation hits)...")
    model.load_state_dict(torch.load(MODELS / "recovery_policy_best.pth"))
    model.eval()
    with torch.no_grad():
        err = (model(xva) - yva).numpy()
    rmse = np.sqrt((err ** 2).mean(axis=0))
    val_ep = ep[is_val]
    per_type = {}
    print(f"   {'type':<6} {'samples':>8} | {'roll rad':>8} {'pitch rad':>9} {'thr x hover':>11}")
    for k in ["flip", "bird", "spin"]:
        m = np.array([kinds.get(int(h)) == k for h in val_ep])
        if m.any():
            r = np.sqrt((err[m] ** 2).mean(axis=0))
            per_type[k] = {"samples": int(m.sum()), **{n: float(v) for n, v in zip(ACT_NAMES, r)}}
            print(f"   {k:<6} {int(m.sum()):>8} | {r[0]:8.3f} {r[1]:9.3f} {r[2]:11.3f}")
    print(f"   {'ALL':<6} {len(err):>8} | {rmse[0]:8.3f} {rmse[1]:9.3f} {rmse[2]:11.3f}")

    print("\n[6/6] Transfer check (BC -> SB3 PPO)...")
    diff = transfer_check(model, Xn[is_val], MODELS)

    info = {
        "teacher": "attitude-rule", "dataset": args.demos, "collection": collect_info,
        "samples": int(len(X)), "hits": int(len(eps)), "split": "by hit 90/10",
        "train_samples": int(np.sum(~is_val)), "val_samples": int(np.sum(is_val)),
        "architecture": "13-256-256-128-3, ReLU", "params": int(n_params),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "best_val_loss": best, "best_epoch": best_ep,
        "val_rmse": {n: float(v) for n, v in zip(ACT_NAMES, rmse)}, "val_rmse_per_type": per_type,
        "transfer_max_action_diff": diff, "training_log": str(csv_path), "trained_at": stamp,
    }
    with open(MODELS / "model_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best:.5f} (epoch {best_ep})")
    print(f"Validation RMSE: roll {rmse[0]:.3f} rad | pitch {rmse[1]:.3f} rad | thrust {rmse[2]:.3f} x hover")
    print(f"Training time: {(time.time() - t0) / 60:.1f} min")
    print(f"Model:   {MODELS / 'recovery_policy_best.pth'}  +  {MODELS / 'stage3_ppo_init.zip'}")
    print(f"Log:     {csv_path}")
    print(f"Plot:    {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", default=str(Path("demos") / "recovery_demos.npz"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
