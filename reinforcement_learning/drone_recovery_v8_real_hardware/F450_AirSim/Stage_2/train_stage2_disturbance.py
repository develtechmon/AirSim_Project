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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from drone_hover_disturbance_env import DroneHoverDisturbanceEnv, run_folder, stage1_folder

HIDDEN = [256, 256, 128]
CSV_COLS = ["episode", "timesteps", "steps", "return", "mean_dist_m", "max_dist_m", "end_dist_m",
            "rmse_m", "hold_pct", "mean_wind_mps", "max_wind_mps", "reason", "elapsed_min",
            "mean_hdist_m", "max_hdist_m"]


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
               max_grad_norm=0.5, target_kl=0.05, seed=args.seed, device="cpu", verbose=1,
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
            print(f"   (critic warm-up: actor frozen until {self.steps:,} steps)")

    def _on_rollout_end(self):
        if self.frozen and self.num_timesteps >= self.steps:
            for q in self._params():
                q.requires_grad_(True)
            self.frozen = False
            print(f"   (critic warm-up done at {self.num_timesteps:,} steps - actor now learning)")

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
                   round(info["ep_max_wind"], 2), info["reason"], round((time.time() - self.t0) / 60, 1),
                   round(info["ep_mean_hdist"], 3), round(info["ep_max_hdist"], 3)]
            self.w.writerow(row)
            self.f.flush()
            self.rows.append(dict(zip(CSV_COLS, [str(v) for v in row])))
            counted = [r for r in self.rows if r["reason"] != "update_pause"]
            if row[11] != "update_pause":
                last20 = counted[-20:]
                self.logger.record("stage2/success_rate_last20_pct",
                                   100.0 * sum(r["reason"] == "timeout" for r in last20) / len(last20))
                self.logger.record("stage2/distance_last20_m",
                                   float(np.mean([float(r["mean_hdist_m"]) for r in last20])))
                self.logger.record("stage2/return_last20", float(np.mean([float(r["return"]) for r in last20])))
            if row[11] != "update_pause" and len(counted) % 10 == 0:
                last = counted[-10:]
                ok = sum(r["reason"] == "timeout" for r in last)
                print("\n" + "=" * 70)
                print(f"📊 EPISODE {len(counted)}")
                print("=" * 70)
                print("   Last 10 Episodes:")
                print(f"      Avg Return: {np.mean([float(r['return']) for r in last]):.1f}")
                print(f"      Avg Length: {np.mean([int(r['steps']) for r in last]):.1f} steps")
                print(f"      Success Rate: {ok * 10}% (reached timeout)")
                print(f"      Avg Wind: {np.mean([float(r['mean_wind_mps']) for r in last]):.2f} m/s")
                print("=" * 70 + "\n")
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
    ax[0].set_title(label or "Stage 2 PPO training (wind 0-20 m/s)")
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=130)
    except OSError as e:   # e.g. the PNG is open in a Windows viewer - never stop training for a picture
        print(f"   (learning curve not updated: {e} - close the picture; it is redrawn at the next rollout)")
    finally:
        plt.close(fig)


def has_progress_bar():
    try:
        import rich  # noqa: F401
        import tqdm  # noqa: F401
        return True
    except ImportError:
        return False


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

    hours = args.timesteps / 20 / 3600 * 1.2
    print("\n" + "=" * 70)
    print("🌬️  STAGE 2: DISTURBANCE RECOVERY TRAINING")
    print("=" * 70)
    print("Training drone to handle wind while hovering at 2m altitude")
    if args.init == "warm":
        print("Starting from Stage 1 hover policy")
    else:
        print("Starting from RANDOM weights (cold start - comparison run)")
    print(f"Expected training time: {hours:.1f} hours")
    print("=" * 70 + "\n")

    ckpt = latest_checkpoint(ckpt_dir) if args.resume else None
    if args.resume and ckpt is None:
        raise SystemExit(f"--resume: no checkpoint in {ckpt_dir}")
    print(f"[1/5] Loading Stage 1 hover policy: {s1_zip}")
    for p in (s1_zip, norm_path):
        if not p.exists():
            print(f"   ❌ ERROR: Stage 1 model not found at {p}")
            print("   💡 Make sure you've trained Stage 1 first, or specify the folder:")
            print("      --stage1-run ..\\Stage_1\\runs\\run1")
            raise SystemExit(1)
    print("   ✅ Stage 1 policy loaded")

    env = DummyVecEnv([lambda: Monitor(DroneHoverDisturbanceEnv(norm_path, max_steps=args.max_steps,
                                                         wind_strength=args.wind_strength, seed=args.seed))])
    base_env = env.envs[0].unwrapped   # Monitor adds rollout/ep_rew_mean to TensorBoard

    print("\n[2/5] Creating PPO model...")
    transfer_diff, t0_min, info = None, 0.0, {}
    if ckpt:
        model = PPO.load(str(ckpt), env=env, device="cpu", tensorboard_log=tb_or_none(log_dir / "tb"))
        model.verbose = 1
        info = json.load(open(info_path)) if info_path.exists() else {}
        transfer_diff = info.get("transfer_max_action_diff")
        rows = read_rows(csv_path) if csv_path.exists() else []
        keep = [r for r in rows if int(r["timesteps"]) <= model.num_timesteps]
        if len(keep) < len(rows):   # episodes after the checkpoint belong to a policy that is discarded
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_COLS, restval="", extrasaction="ignore")
                w.writeheader()
                w.writerows(keep)
            print(f"   (removed {len(rows) - len(keep)} episodes logged after the checkpoint - they are re-flown)")
        t0_min = float(keep[-1]["elapsed_min"]) if keep else 0.0
        print(f"   ✅ PPO model resumed from {ckpt.name} ({model.num_timesteps:,} steps)")
        print("\n[3/5] Loading pretrained weights into PPO actor...")
        print("   ✅ Already inside the checkpoint")
    else:
        model = make_model(env, args, log_dir / "tb")
        print("   ✅ PPO model created")
        print("\n[3/5] Loading pretrained weights into PPO actor...")
        if args.init == "warm":
            transfer_diff = load_warm(model, s1_zip)
            print("   ✅ Pretrained weights loaded into actor network")
            print("   💡 PPO will start from Stage 1 hover skill!")
            print("   📈 Only needs to learn wind compensation")
        else:
            print("   ⚠️  Cold start: no pretrained weights (random actor)")
        info = {"init": args.init, "run": run_dir.name, "stage1_run": str(s1),
                "stage1_model": str(s1_zip) if args.init == "warm" else None,
                "transfer_max_action_diff": transfer_diff, "settings": vars(args),
                "ppo": {"lr": args.lr, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99,
                        "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "target_kl": 0.05,
                        "log_std_init": args.log_std_init, "net_arch": "13-256-256-128-3 ReLU (pi and vf)",
                        "critic_warmup_steps": args.critic_warmup},
                "env": {"wind": f"0-{args.wind_strength} m/s random direction, vertical x0.3, new gust every 1-3 s",
                        "episode_steps": args.max_steps, "hz": 20, "fail_dist_m": 4.0,
                        "reward": "+1 - dist - 0.1*speed - 0.01*|a|^2, -50 on failure",
                        "obs_norm": str(norm_path)},
                "started_at": datetime.datetime.now().isoformat(timespec="seconds")}
        json.dump(info, open(info_path, "w"), indent=2, default=str)

    print("\n[4/5] Creating disturbance environment...")
    print("   Target altitude: 2m")
    print(f"   Wind strength: 0-{args.wind_strength} m/s (new gust every 1-3 s)")
    print(f"   Episode length: {args.max_steps} steps ({args.max_steps * 0.05:.1f}s)")
    print("   ✅ Environment created with wind disturbances")

    logger = EpisodeLogger(csv_path, log_dir / "learning_curve.png", t0_min)
    callbacks = CallbackList([
        ActorFreeze(args.critic_warmup), logger,
        CheckpointCallback(save_freq=25000, save_path=str(ckpt_dir), name_prefix="disturbance_policy")])
    remaining = args.timesteps - model.num_timesteps
    if remaining <= 0:
        raise SystemExit(f"Already at {model.num_timesteps:,} steps (target {args.timesteps:,}) - nothing to do. "
                         "Raise --timesteps to train longer.")
    print("\n[5/5] Starting PPO training...")
    print(f"   Total timesteps: {args.timesteps:,}")
    print(f"   Learning rate: {args.lr}")
    print("   Checkpoints: Every 25,000 steps")
    print(f"   Estimated time: {hours:.1f} hours")
    print()
    print("=" * 70)
    print("🚀 TRAINING STARTED")
    print("=" * 70)
    print("Watch for episode statistics every 10 episodes...")
    print("Model will learn to compensate for wind disturbances!")
    print("=" * 70 + "\n")
    status = "running"
    try:
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=not ckpt,
                    tb_log_name="tb", progress_bar=has_progress_bar())
        model.save(str(final_path))
        status = "complete"
    except KeyboardInterrupt:
        status = "interrupted"
        p = ckpt_dir / f"disturbance_policy_{model.num_timesteps}_steps.zip"
        model.save(str(p))
        print("\n⚠️ Training interrupted by user")
        print(f"💾 Model saved at interruption point: {p}")
        print(f"   Continue with: py .\\train_stage2_disturbance.py --init {args.init} --resume")
    except Exception:
        status = "crashed"
        p = ckpt_dir / f"disturbance_policy_{model.num_timesteps}_steps.zip"
        try:
            model.save(str(p))
            print(f"\n❌ Training stopped by an error (details below). Model saved at that point: {p}")
        except Exception:
            print("\n❌ Training stopped by an error (details below). Last automatic checkpoint is kept.")
        print(f"   Fix the cause, then continue with: py .\\train_stage2_disturbance.py --init {args.init} --resume")
        raise
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
        if status == "complete":
            print("\n" + "=" * 70)
            print("✅ TRAINING COMPLETE!")
            print("=" * 70)
            print("\n💾 Model saved:")
            print(f"   - {final_path}")
            print(f"   - {norm_path} (input scaling, from Stage 1)")
            print("\n📊 Training Statistics:")
            print(f"   Total episodes: {len(counted)}")
            if last:
                print(f"   Last {len(last)} episodes:")
                print(f"      Avg return: {info['last50_return']:.1f}")
                print(f"      Avg length: {np.mean([int(r['steps']) for r in last]):.1f} steps")
                print(f"      Success rate: {info['last50_success_pct']:.0f}%")
            print("\n✅ Next step: Test the model!")
            print("   py .\\test_stage2_policy.py")
            print("=" * 70 + "\n")
        base_env.land()
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="warm", choices=["warm", "cold"],
                    help="warm = start from Stage 1 (default, like your original); cold = random start, comparison only")
    ap.add_argument("--timesteps", type=int, default=500000)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--wind-strength", type=float, default=20.0, help="AirSim gust max (20 -> ~14 deg lean)")
    ap.add_argument("--lr", type=float, default=3e-5, help="same as the original Stage 2")
    ap.add_argument("--log-std-init", type=float, default=-1.2, help="exploration std = exp(value); -1.2 -> 0.30 m/s")
    ap.add_argument("--critic-warmup", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage1-run", default=None, help="Stage 1 run folder (default: newest ../Stage_1/runs/run<N>)")
    ap.add_argument("--run", default=None, help="Stage 2 run number (default: latest)")
    ap.add_argument("--new-run", action="store_true", help="start a new Stage 2 run folder")
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint")
    main(ap.parse_args())
