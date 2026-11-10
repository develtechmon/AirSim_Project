"""
STAGE 2: PPO TRAINING WITH WIND (ArduPilot SITL + AirSim, 13 observations)
==========================================================================
Same job as your AirSim train_stage2_disturbance.py, with the transfer bug fixed.

Two ways to start (same settings otherwise, so the comparison is fair):
    --init warm   PPO starts from the Stage 1 network (stage1_ppo_init.zip).
                  Checked at start: PPO's action must equal Stage 1's action exactly (PASS).
    --init cold   PPO starts from random weights (the "cold start" baseline).

What was wrong in the original (and is fixed here):
    1. Weights were loaded under names that do not exist in PPO, with strict=False,
       so NOTHING was loaded - the original Stage 2 was really a cold start.
    2. PPO used Tanh, Stage 1 used ReLU.               -> here: ReLU, same layers
    3. VecNormalize learned new input scaling.          -> here: Stage 1's obs_norm.npz
    4. Exploration noise std 1.0 m/s, entropy bonus.    -> here: std 0.3 m/s, no entropy bonus
    5. Very large rewards (value loss ~1e6).            -> here: about -3..+1 per step

Critic warm-up: for the first 20,000 steps only the value network learns (the actor is frozen),
so the random critic cannot damage the Stage 1 behaviour at the start. Used for warm AND cold.

Usage (AirSim + SITL in GPS mode, run from the stage_2 folder):
    py .\\train_stage2_disturbance.py --init warm                 (first run -> runs\\run1)
    py .\\train_stage2_disturbance.py --init cold                 (same run folder)
    py .\\train_stage2_disturbance.py --init warm --resume        (continue after a stop)
    py .\\train_stage2_disturbance.py --init warm --new-run       (new experiment -> next run)
    tensorboard --logdir runs

Saves in runs\\run<N>\\:
    models\\<init>\\hover_disturbance_policy.zip      final network
    models\\<init>\\checkpoints\\                      every 25,000 steps (used by --resume)
    models\\<init>\\training_info.json                 settings, transfer check, results
    logs\\stage2\\<init>\\episodes.csv                  one line per episode (written live)
    logs\\stage2\\<init>\\learning_curve.png            return / success / distance vs steps
    logs\\stage2\\<init>\\tb\\                           TensorBoard
"""

import argparse
import csv
import datetime
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from drone_hover_disturbance_env import DroneHoverDisturbanceEnv, run_folder, stage1_folder

HIDDEN = [256, 256, 128]
CSV_COLS = ["episode", "timesteps", "steps", "return", "mean_dist_m", "max_dist_m", "end_dist_m",
            "rmse_m", "hold_pct", "mean_wind_mps", "max_wind_mps", "reason", "elapsed_min"]


def tb_or_none(tb_dir):
    try:
        import tensorboard  # noqa: F401
        return str(tb_dir)
    except ImportError:
        print("   (TensorBoard not installed: pip install tensorboard - CSV log and plots still saved)")
        return None


def make_model(env, args, tb_dir):
    return PPO("MlpPolicy", env, learning_rate=args.lr, n_steps=2048, batch_size=64, n_epochs=10,
               gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
               max_grad_norm=0.5, target_kl=0.02, seed=args.seed, device="cpu", verbose=0,
               tensorboard_log=tb_or_none(tb_dir),
               policy_kwargs=dict(net_arch=dict(pi=HIDDEN, vf=HIDDEN), activation_fn=nn.ReLU,
                                  log_std_init=args.log_std_init))


def load_warm(model, stage1_zip):
    """Copy the Stage 1 actor into PPO and prove PPO acts exactly like Stage 1."""
    s1 = PPO.load(str(stage1_zip), device="cpu")
    actor = {k: v for k, v in s1.policy.state_dict().items()
             if k.startswith("mlp_extractor.policy_net") or k.startswith("action_net")}
    res = model.policy.load_state_dict(actor, strict=False)
    bad = [k for k in res.missing_keys
           if k.startswith("mlp_extractor.policy_net") or k.startswith("action_net")]
    if res.unexpected_keys or bad:
        raise RuntimeError(f"WARM START FAILED - unexpected {res.unexpected_keys}, not loaded {bad}")
    x = torch.as_tensor(np.random.RandomState(0).uniform(-3, 3, (2000, 13)).astype(np.float32))
    with torch.no_grad():
        a_new = model.policy.get_distribution(x).distribution.mean
        a_s1 = s1.policy.get_distribution(x).distribution.mean
    diff = float((a_new - a_s1).abs().max())
    print(f"   Loaded {len(actor)} actor tensors from {stage1_zip}")
    print(f"   max |PPO - Stage 1| action difference: {diff:.2e}  -> {'PASS' if diff < 1e-5 else 'FAIL'}")
    if diff >= 1e-5:
        raise RuntimeError("WARM START FAILED - actions differ")
    return diff


class ActorFreeze(BaseCallback):
    """Critic warm-up: actor frozen for the first N steps, then everything learns."""

    def __init__(self, steps):
        super().__init__()
        self.steps, self.frozen = steps, False

    def _params(self):
        p = self.model.policy
        return list(p.mlp_extractor.policy_net.parameters()) + list(p.action_net.parameters()) + [p.log_std]

    def _on_training_start(self):
        if self.num_timesteps < self.steps:
            for q in self._params():
                q.requires_grad_(False)
            self.frozen = True
            print(f"   Critic warm-up: actor frozen until {self.steps:,} steps")

    def _on_rollout_end(self):
        if self.frozen and self.num_timesteps >= self.steps:
            for q in self._params():
                q.requires_grad_(True)
            self.frozen = False
            print(f"\n   Critic warm-up done at {self.num_timesteps:,} steps - actor now learning\n")

    def _on_step(self):
        return True


class EpisodeLogger(BaseCallback):
    """One CSV line per episode (written live) + a printout every 10 episodes."""

    def __init__(self, csv_path, plot_path, t0_min=0.0):
        super().__init__()
        self.csv_path, self.plot_path = csv_path, plot_path
        new = not csv_path.exists()
        self.f = open(csv_path, "a", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(CSV_COLS)
        self.rows = read_rows(csv_path) if not new else []
        self.t0 = time.time() - 60.0 * t0_min

    def _on_step(self):
        for info in self.locals["infos"]:
            if not info.get("episode_done"):
                continue
            row = [len(self.rows) + 1, self.num_timesteps, info["ep_steps"], round(info["ep_return"], 2),
                   round(info["ep_mean_dist"], 3), round(info["ep_max_dist"], 3), round(info["ep_end_dist"], 3),
                   round(info["ep_rmse"], 3), round(info["ep_hold_pct"], 1), round(info["ep_mean_wind"], 2),
                   round(info["ep_max_wind"], 2), info["reason"], round((time.time() - self.t0) / 60, 1)]
            self.w.writerow(row)
            self.f.flush()
            self.rows.append(dict(zip(CSV_COLS, [str(v) for v in row])))
            counted = [r for r in self.rows if r["reason"] != "update_pause"]
            if row[11] != "update_pause" and len(counted) % 10 == 0:
                last = counted[-10:]
                ok = sum(r["reason"] == "timeout" for r in last)
                print(f"Episode {len(counted):5d} | steps {self.num_timesteps:8,} | "
                      f"return {np.mean([float(r['return']) for r in last]):7.1f} | "
                      f"success {ok * 10:3d}% | mean dist {np.mean([float(r['mean_dist_m']) for r in last]):.2f} m | "
                      f"wind {np.mean([float(r['mean_wind_mps']) for r in last]):.1f} m/s | "
                      f"{(time.time() - self.t0) / 60:.0f} min")
        return True

    def _on_rollout_end(self):
        save_learning_curve(self.rows, self.plot_path)

    def close(self):
        self.f.close()


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def save_learning_curve(rows, path, label=None):
    rows = [r for r in rows if r["reason"] != "update_pause"]
    if len(rows) < 2:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t = np.array([int(r["timesteps"]) for r in rows])
    ret = np.array([float(r["return"]) for r in rows])
    ok = np.array([100.0 * (r["reason"] == "timeout") for r in rows])
    dist = np.array([float(r["mean_dist_m"]) for r in rows])
    k = min(20, len(rows))
    roll = lambda v: np.convolve(v, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for a, v, name in ((ax[0], ret, "episode return"), (ax[1], ok, "success (%)"),
                       (ax[2], dist, "mean distance (m)")):
        a.plot(t, v, alpha=0.25, linewidth=0.8)
        a.plot(t[k - 1:], roll(v), linewidth=1.8, label=f"rolling mean of {k}")
        a.set_ylabel(name)
        a.grid(alpha=0.3)
    ax[0].legend()
    ax[2].set_xlabel("timesteps")
    ax[0].set_title(label or "Stage 2 PPO training (wind 0-5 m/s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def latest_checkpoint(folder):
    best, best_n = None, -1
    for p in Path(folder).glob("disturbance_policy_*_steps.zip"):
        m = re.search(r"_(\d+)_steps\.zip$", p.name)
        if m and int(m.group(1)) > best_n:
            best, best_n = p, int(m.group(1))
    return best


def main(args):
    run_dir = run_folder(args.run, new=args.new_run)
    model_dir = run_dir / "models" / args.init
    ckpt_dir = model_dir / "checkpoints"
    log_dir = run_dir / "logs" / "stage2" / args.init
    csv_path = log_dir / "episodes.csv"
    final_path = model_dir / "hover_disturbance_policy.zip"
    info_path = model_dir / "training_info.json"
    has_rows = csv_path.exists() and len(read_rows(csv_path)) > 0
    if not args.resume and (final_path.exists() or has_rows):
        raise SystemExit(f"'{args.init}' already has results in {run_dir}. "
                         "Use --resume to continue it, or --new-run for a new experiment.")
    for d in (ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    s1 = stage1_folder(args.stage1_run)
    norm_path = s1 / "models" / "obs_norm.npz"
    s1_zip = s1 / "models" / "stage1_ppo_init.zip"

    print("\n" + "=" * 70)
    print(f"STAGE 2: PPO WITH WIND - {args.init.upper()} START - {run_dir.name.upper()}")
    print("=" * 70)
    print(f"Run folder:  {run_dir}")
    print(f"Stage 1:     {s1}")
    print(f"Timesteps:   {args.timesteps:,}  (~{args.timesteps / 20 / 3600 * 1.2:.1f} h with resets)")
    print(f"Wind:        0-{args.wind_strength} m/s, new gust every 1-3 s | episode {args.max_steps} steps")
    print("=" * 70 + "\n")

    print("[1/3] Connecting to AirSim + ArduPilot...")
    env = DummyVecEnv([lambda: DroneHoverDisturbanceEnv(norm_path, max_steps=args.max_steps,
                                                         wind_strength=args.wind_strength, seed=args.seed)])
    base_env = env.envs[0]

    print("\n[2/3] Creating PPO...")
    transfer_diff, t0_min, info = None, 0.0, {}
    ckpt = latest_checkpoint(ckpt_dir) if args.resume else None
    if args.resume and ckpt is None:
        raise SystemExit(f"--resume: no checkpoint in {ckpt_dir}")
    if ckpt:
        model = PPO.load(str(ckpt), env=env, device="cpu", tensorboard_log=tb_or_none(log_dir / "tb"))
        info = json.load(open(info_path)) if info_path.exists() else {}
        transfer_diff = info.get("transfer_max_action_diff")
        rows = read_rows(csv_path) if csv_path.exists() else []
        t0_min = float(rows[-1]["elapsed_min"]) if rows else 0.0
        print(f"   Resumed from {ckpt.name} at {model.num_timesteps:,} steps")
    else:
        model = make_model(env, args, log_dir / "tb")
        if args.init == "warm":
            transfer_diff = load_warm(model, s1_zip)
        else:
            print("   Random weights (cold start)")
        info = {"init": args.init, "run": run_dir.name, "stage1_run": str(s1),
                "stage1_model": str(s1_zip) if args.init == "warm" else None,
                "transfer_max_action_diff": transfer_diff, "settings": vars(args),
                "ppo": {"lr": args.lr, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99,
                        "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "target_kl": 0.02,
                        "log_std_init": args.log_std_init, "net_arch": "13-256-256-128-3 ReLU (pi and vf)",
                        "critic_warmup_steps": args.critic_warmup},
                "env": {"wind": f"0-{args.wind_strength} m/s random direction, vertical x0.3, change 1-3 s",
                        "episode_steps": args.max_steps, "hz": 20, "fail_dist_m": 4.0,
                        "reward": "+1 - dist - 0.1*speed - 0.01*|a|^2, -50 on failure",
                        "obs_norm": str(norm_path)},
                "started_at": datetime.datetime.now().isoformat(timespec="seconds")}
        json.dump(info, open(info_path, "w"), indent=2, default=str)

    logger = EpisodeLogger(csv_path, log_dir / "learning_curve.png", t0_min)
    callbacks = CallbackList([
        ActorFreeze(args.critic_warmup), logger,
        CheckpointCallback(save_freq=25000, save_path=str(ckpt_dir), name_prefix="disturbance_policy")])
    remaining = args.timesteps - model.num_timesteps
    if remaining <= 0:
        raise SystemExit(f"Already at {model.num_timesteps:,} steps (target {args.timesteps:,}) - nothing to do. "
                         "Raise --timesteps to train longer.")
    print(f"\n[3/3] Training {remaining:,} steps... (Ctrl+C saves a checkpoint; --resume continues)\n")
    status = "complete"
    try:
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=not ckpt,
                    tb_log_name="tb", progress_bar=False)
        model.save(str(final_path))
    except KeyboardInterrupt:
        status = "interrupted"
        p = ckpt_dir / f"disturbance_policy_{model.num_timesteps}_steps.zip"
        model.save(str(p))
        print(f"\nStopped by user - saved {p.name}. Continue with: --init {args.init} --resume")
    finally:
        logger.close()
        rows = read_rows(csv_path) if csv_path.exists() else []
        save_learning_curve(rows, log_dir / "learning_curve.png",
                            f"Stage 2 PPO, {args.init} start (wind 0-{args.wind_strength} m/s)")
        counted = [r for r in rows if r["reason"] != "update_pause"]
        last = counted[-50:]
        info.update({"status": status, "timesteps_done": int(model.num_timesteps),
                     "episodes": len(counted),
                     "last50_success_pct": 100.0 * sum(r["reason"] == "timeout" for r in last) / max(1, len(last)),
                     "last50_mean_dist_m": float(np.mean([float(r["mean_dist_m"]) for r in last])) if last else None,
                     "last50_return": float(np.mean([float(r["return"]) for r in last])) if last else None,
                     "finished_at": datetime.datetime.now().isoformat(timespec="seconds")})
        json.dump(info, open(info_path, "w"), indent=2, default=str)
        print("\n" + "=" * 70)
        print(f"TRAINING {status.upper()} - {args.init} start - {run_dir.name}")
        print("=" * 70)
        print(f"Timesteps: {model.num_timesteps:,} | episodes: {len(counted)}")
        if last:
            print(f"Last {len(last)} episodes: success {info['last50_success_pct']:.0f}% | "
                  f"mean dist {info['last50_mean_dist_m']:.2f} m | return {info['last50_return']:.1f}")
        print(f"Model: {final_path if status == 'complete' else ckpt_dir}")
        print(f"Log:   {csv_path}")
        print("=" * 70)
        base_env.land()
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, choices=["warm", "cold"])
    ap.add_argument("--timesteps", type=int, default=500000)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--wind-strength", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-std-init", type=float, default=-1.2, help="exploration std = exp(value); -1.2 -> 0.30 m/s")
    ap.add_argument("--critic-warmup", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage1-run", default=None, help="Stage 1 run folder (default: newest ../stage_1/runs/run<N>)")
    ap.add_argument("--run", default=None, help="Stage 2 run number (default: latest)")
    ap.add_argument("--new-run", action="store_true", help="start a new Stage 2 run folder")
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint")
    main(ap.parse_args())
