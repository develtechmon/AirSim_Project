"""
PERFORMANCE-GATED CURRICULUM TRAINING - WITH COMPREHENSIVE LOGGING (ArduPilot SITL + AirSim)
============================================================================================
Same flow, same printout and same files as your AirSim train_stage_3_gated_curriculum.py.
Only advances to the next level when the current level is MASTERED:
    Easy (0.7-0.9):   80% recovery required
    Medium (0.9-1.1): 70% recovery required
    Hard (1.1-1.5):   60% recovery target (PhD complete)

Generates (same names as AirSim):
    logs/stage3/gated_training_TIMESTAMP_episodes.csv     one line per episode (written live)
    logs/stage3/gated_training_TIMESTAMP_summary.json
    logs/stage3/gated_training_TIMESTAMP_curriculum.json
    logs/gated_curriculum/PPO_1/events.out.tfevents.*     TensorBoard
    models/stage3_checkpoints/curriculum_levels/level_0_EASY_mastered.zip (+ _metadata.json)
    models/stage3_checkpoints/curriculum_levels/level_1_MEDIUM_mastered.zip (+ _metadata.json)
    models/stage3_checkpoints/gated_checkpoints/gated_curriculum_<steps>_steps.zip  (every 50k)
    models/stage3_checkpoints/gated_curriculum_policy.zip  (final)
Then: py .\\analyze_training_logs.py --output-dir .\\thesis_figures
      -> training_learning_curves.pdf/png, curriculum_progression.pdf/png,
         training_statistics.json, analysis_results\\table_5_1 / table_5_2 .tex

Differences from AirSim (see the environment file for details):
    - starts from YOUR Stage 2 network (ArduPilot), input scaling = Stage 1 obs_norm.npz
      (no VecNormalize, so there is no *_vecnormalize.pkl - the scaling is fixed)
    - critic warm-up: the actor is frozen for the first 20,000 steps so the critic learns
      the new reward first (same as Stage 2)
    - Ctrl+C or a crash saves a checkpoint at that exact step; --resume continues it,
      including the curriculum level (rebuilt from the episode log)
    - episodes cut by a PPO update ('update_pause') are not logged and not counted

Usage (in Stage_3, AirSim + SITL in GPS mode running):
    py .\\train_stage_3_gated_curriculum.py
    py .\\train_stage_3_gated_curriculum.py --resume
"""

import argparse
import csv
import datetime
import json
import re
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv

from drone_flip_recovery_env_injector_gated import HZ, LEVEL_NAMES, DroneFlipRecoveryEnv

LEVEL_LABELS = ["EASY (0.7-0.9)", "MEDIUM (0.9-1.1)", "HARD (1.1-1.5)"]
MODELS = Path("./models/stage3_checkpoints/")
LEVEL_DIR = MODELS / "curriculum_levels"
CKPT_DIR = MODELS / "gated_checkpoints"
LOG_DIR = Path("./logs/stage3/")
TB_DIR = "./logs/gated_curriculum/"
CSV_COLS = ["episode", "timestamp", "elapsed_time_s", "curriculum_level", "episode_reward", "episode_length",
            "tumble_initiated", "tumble_recovered", "recovery_steps", "disturbance_intensity", "disturbance_type",
            "rolling_10_reward", "rolling_10_recovery_rate", "rolling_50_recovery_rate",
            # extra (ArduPilot): not in the AirSim log, used for the thesis tables
            "timesteps", "reason", "recovery_time_s", "min_altitude", "impact_turns", "impact_turns_planned",
            "impact_peak_rate", "max_angular_velocity", "final_distance"]


def latest_checkpoint():
    best, best_n = None, -1
    for p in CKPT_DIR.glob("gated_curriculum_*_steps.zip"):
        m = re.search(r"_(\d+)_steps\.zip$", p.name)
        if m and int(m.group(1)) > best_n:
            best, best_n = p, int(m.group(1))
    return best


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class ActorFreeze(BaseCallback):
    """Critic warm-up: actor frozen for the first N steps (same as Stage 2)."""

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


class GatedCurriculumCallbackWithLogging(BaseCallback):
    """Same logging as AirSim: CSV per episode, printout every 10 episodes, auto-save on advancement."""

    def __init__(self, base_env, csv_path, prefix, old_rows=None, t0_s=0.0):
        super().__init__()
        self.base_env = base_env
        self.csv_log_path, self.log_prefix = csv_path, prefix
        self.start_time = time.time() - t0_s
        self.episode_rewards, self.episode_lengths, self.curriculum_levels = [], [], []
        self.recovery_successes, self.recovery_times, self.intensities = [], [], []
        self.recovery_level, self.level_achievements, self.update_pauses = [], {}, 0
        new = old_rows is None
        for r in old_rows or []:
            self._remember(float(r["episode_reward"]), int(float(r["episode_length"])), int(r["curriculum_level"]),
                           int(r["tumble_initiated"]), int(r["tumble_recovered"]), int(float(r["recovery_steps"])),
                           float(r["disturbance_intensity"]), float(r["elapsed_time_s"]), float(r["timestamp"]))
        self.episode_count = len(self.episode_rewards)
        self.csv_file = open(csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if new:
            self.csv_writer.writerow(CSV_COLS)
            self.csv_file.flush()
        print("\n📝 LOGGING ENABLED:")
        print(f"   CSV Log: {self.csv_log_path}")
        print(f"   Summary will be saved to: {LOG_DIR / f'{self.log_prefix}_summary.json'}")
        print("   All logs saved to: ./logs/stage3/\n")

    def _remember(self, ret, length, level, initiated, recovered, rec_steps, intensity, elapsed, stamp):
        self.episode_rewards.append(ret)
        self.episode_lengths.append(length)
        self.curriculum_levels.append(level)
        if level not in self.level_achievements:
            self.level_achievements[level] = {"reached_at_episode": len(self.episode_rewards),
                                              "elapsed_time": elapsed, "timestamp": stamp}
        if initiated:
            self.recovery_successes.append(recovered)
            self.recovery_level.append(level)
            self.intensities.append(intensity)
            if recovered:
                self.recovery_times.append(rec_steps)

    def _on_step(self):
        for info in self.locals["infos"]:
            if not info.get("episode_done"):
                continue
            if info["reason"] == "update_pause":
                self.update_pauses += 1
                continue
            self.episode_count += 1
            now = time.time()
            elapsed = now - self.start_time
            level = info["curriculum_level"]
            # the level the episode was FLOWN at (advancement happens at its end)
            if self.base_env.level_advanced:
                level = self.base_env.advancement_info["old_level"]
            ini, rec = int(info["tumble_initiated"]), int(info["tumble_recovered"])
            self._remember(info["episode_return"], info["episode_steps"], level, ini, rec,
                           info["recovery_steps"], info["disturbance_intensity"], elapsed, now)
            r10 = float(np.mean(self.episode_rewards[-10:]))
            rr10 = float(np.mean(self.recovery_successes[-10:]) * 100) if len(self.recovery_successes) >= 10 else 0.0
            rr50 = float(np.mean(self.recovery_successes[-50:]) * 100) if len(self.recovery_successes) >= 50 else 0.0
            self.csv_writer.writerow([
                self.episode_count, now, round(elapsed, 1), level, round(info["episode_return"], 3),
                info["episode_steps"], ini, rec, info["recovery_steps"], round(info["disturbance_intensity"], 3),
                info["disturbance_type"], round(r10, 3), round(rr10, 1), round(rr50, 1),
                self.num_timesteps, info["reason"], round(info["recovery_time_s"], 2),
                round(info["min_altitude"], 2), round(info["impact_turns"], 2), round(info["impact_turns_planned"], 2),
                round(info["impact_peak_rate"], 2), round(info["max_angular_velocity"], 2),
                round(info["final_distance"], 2)])
            self.csv_file.flush()                       # never lose data on a crash
            self.logger.record("stage3/recovery_rate_last50_pct", rr50)
            self.logger.record("stage3/return_last10", r10)
            self.logger.record("stage3/curriculum_level", level)
            if self.base_env.level_advanced:
                self._save_advancement_model(self.base_env.advancement_info)
                self.base_env.level_advanced = False
            if self.episode_count % 10 == 0:
                self._print_block(elapsed)
        return True

    def _print_block(self, elapsed):
        recent_rec = self.recovery_successes[-10:]
        rate = float(np.mean(recent_rec) * 100) if recent_rec else 0.0
        avg_int = float(np.mean(self.intensities[-10:])) if self.intensities else 0.0
        avg_rt = float(np.mean(self.recovery_times[-10:])) if self.recovery_times else 0.0
        lvl = self.base_env.curriculum_level
        print(f"\n{'=' * 70}")
        print(f"📊 EPISODE {self.episode_count} | {elapsed / 3600:.1f}h elapsed")
        print(f"{'=' * 70}")
        print(f"   🎓 Curriculum Level: {lvl} - {LEVEL_LABELS[lvl]}")
        print("   Last 10 Episodes:")
        print(f"      Avg Return: {np.mean(self.episode_rewards[-10:]):.1f}")
        print(f"      Avg Length: {np.mean(self.episode_lengths[-10:]):.1f} steps")
        print("   Recovery Performance:")
        print(f"      Recovery Rate: {rate:.0f}%")
        print(f"      Avg Intensity: {avg_int:.2f}x")
        if avg_rt > 0:
            print(f"      Avg Recovery Time: {avg_rt:.0f} steps ({avg_rt / HZ:.1f}s)")
        target = [80, 70, 60][lvl]
        print(f"   📈 Progress: {rate:.0f}% / {target}% " + ("(PhD target)" if lvl == 2 else "(need to advance)"))
        print(f"{'=' * 70}\n")

    def _save_advancement_model(self, a):
        old = a["old_level"]
        name = f"level_{old}_{LEVEL_NAMES[old]}_mastered"
        path = LEVEL_DIR / name
        print("\n" + "=" * 70)
        print("💾 AUTO-SAVING MODEL - LEVEL MASTERED!")
        print("=" * 70)
        print(f"   Level {old} ({LEVEL_NAMES[old]}) completed")
        print(f"   Recovery rate: {a['recovery_rate'] * 100:.1f}%")
        print(f"   Episode: {self.episode_count}")
        print(f"   Saving to: {path}")
        self.model.save(str(path))
        meta = {"level": old, "level_name": LEVEL_NAMES[old], "recovery_rate": float(a["recovery_rate"]),
                "episode": self.episode_count, "timesteps": int(self.num_timesteps),
                "timestamp": time.time() - self.start_time, "total_episodes": self.episode_count,
                "obs_norm": "Stage 1 obs_norm.npz (fixed scaling, no VecNormalize)"}
        with open(LEVEL_DIR / f"{name}_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"   ✅ Model saved: {path}.zip")
        print(f"   ✅ Metadata saved: {LEVEL_DIR / f'{name}_metadata.json'}")
        print("=" * 70 + "\n")

    def save_final_summary(self):
        hours = (time.time() - self.start_time) / 3600
        last = lambda v, n: v[-n:] if len(v) >= n else v
        summary = {
            "training_info": {"total_episodes": self.episode_count, "total_time_seconds": hours * 3600,
                              "total_time_hours": hours, "episodes_per_hour": self.episode_count / max(hours, 1e-9),
                              "episodes_cut_by_ppo_update": self.update_pauses, "end_time": time.time()},
            "final_performance": {
                "last_10_avg_reward": float(np.mean(last(self.episode_rewards, 10))) if self.episode_rewards else 0,
                "last_50_avg_reward": float(np.mean(last(self.episode_rewards, 50))) if self.episode_rewards else 0,
                "last_10_recovery_rate": float(np.mean(last(self.recovery_successes, 10)) * 100) if self.recovery_successes else 0,
                "last_50_recovery_rate": float(np.mean(last(self.recovery_successes, 50)) * 100) if self.recovery_successes else 0,
                "avg_recovery_time_steps": float(np.mean(last(self.recovery_times, 50))) if self.recovery_times else 0,
                "avg_recovery_time_seconds": float(np.mean(last(self.recovery_times, 50)) / HZ) if self.recovery_times else 0},
            "curriculum_progression": {}, "statistics_by_level": {}}
        for level, d in sorted(self.level_achievements.items()):
            summary["curriculum_progression"][f"level_{level}"] = {
                "level_name": LEVEL_LABELS[level], "reached_at_episode": d["reached_at_episode"],
                "elapsed_time_hours": d["elapsed_time"] / 3600}
        for level in range(3):
            eps = [i for i, lv in enumerate(self.curriculum_levels) if lv == level]
            if eps:
                rec = [s for s, lv in zip(self.recovery_successes, self.recovery_level) if lv == level]
                ints = [x for x, lv in zip(self.intensities, self.recovery_level) if lv == level]
                summary["statistics_by_level"][f"level_{level}_{LEVEL_NAMES[level]}"] = {
                    "episodes_trained": len(eps), "recovery_rate": float(np.mean(rec) * 100) if rec else 0,
                    "avg_reward": float(np.mean([self.episode_rewards[i] for i in eps])),
                    "avg_intensity": float(np.mean(ints)) if ints else 0}
        sp = LOG_DIR / f"{self.log_prefix}_summary.json"
        json.dump(summary, open(sp, "w"), indent=2)
        cp = LOG_DIR / f"{self.log_prefix}_curriculum.json"
        json.dump({"level_achievements": {str(k): v for k, v in self.level_achievements.items()},
                   "curriculum_levels_per_episode": self.curriculum_levels,
                   "recovery_successes_per_episode": self.recovery_successes,
                   "intensities_per_episode": self.intensities}, open(cp, "w"), indent=2)
        print("\n📊 TRAINING LOGS SAVED:")
        print(f"   ✅ Episode log: {self.csv_log_path}")
        print(f"   ✅ Summary: {sp}")
        print(f"   ✅ Curriculum: {cp}")
        return summary

    def close(self):
        if not self.csv_file.closed:
            self.csv_file.close()


def restore_from_log(rows, base_env):
    """--resume: curriculum level, episodes at level and recent recoveries from the episode log."""
    if not rows:
        return
    level = int(rows[-1]["curriculum_level"])
    at_level = []
    for r in reversed(rows):
        if int(r["curriculum_level"]) != level:
            break
        at_level.append(r)
    at_level.reverse()
    recent = [int(r["tumble_recovered"]) for r in at_level if int(r["tumble_initiated"])]
    base_env.set_curriculum_state(level, len(at_level), recent)
    base_env.episode_count = len(rows)
    print(f"   ✅ Curriculum restored: Level {level} ({LEVEL_NAMES[level]}), {len(at_level)} episodes at level, "
          f"recent recovery {np.mean(recent[-50:]) * 100 if recent else 0:.0f}%")


def main(args):
    for d in (LEVEL_DIR, CKPT_DIR, LOG_DIR, Path(TB_DIR)):
        d.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print("🎓 PERFORMANCE-GATED CURRICULUM TRAINING - WITH LOGGING (ArduPilot)")
    print("=" * 70)
    print("Training with automatic difficulty advancement + comprehensive logging!\n")
    print("Progression Rules:")
    print("  Level 0 (EASY):   0.7-0.9x intensity → Need 80% recovery")
    print("  Level 1 (MEDIUM): 0.9-1.1x intensity → Need 70% recovery")
    print("  Level 2 (HARD):   1.1-1.5x intensity → Target 60% recovery\n")
    print("✅ Only advances when current level is MASTERED!")
    print("✅ Saves CSV + JSON logs for PhD analysis!")
    print("=" * 70 + "\n")

    ckpt = latest_checkpoint() if args.resume else None
    if args.resume and ckpt is None:
        raise SystemExit(f"--resume: no checkpoint in {CKPT_DIR}")
    old_rows, csv_path, prefix, t0_s = None, None, None, 0.0
    if ckpt:
        logs = sorted(LOG_DIR.glob("gated_training_*_episodes.csv"))
        if not logs:
            raise SystemExit("--resume: no episode log in logs/stage3/")
        csv_path = logs[-1]
        prefix = csv_path.name.replace("_episodes.csv", "")
        steps = int(re.search(r"_(\d+)_steps\.zip$", ckpt.name).group(1))
        rows = read_rows(csv_path)
        old_rows = [r for r in rows if int(r["timesteps"]) <= steps]
        if len(old_rows) < len(rows):
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_COLS, restval="", extrasaction="ignore")
                w.writeheader()
                w.writerows(old_rows)
            print(f"   (removed {len(rows) - len(old_rows)} episodes logged after the checkpoint - they are re-flown)")
        t0_s = float(old_rows[-1]["elapsed_time_s"]) if old_rows else 0.0
    else:
        if list(LOG_DIR.glob("gated_training_*_episodes.csv")) and not args.new_run:
            raise SystemExit("logs/stage3 already has a training log. Use --resume to continue it, "
                             "or --new-run to start a new training (old files are kept).")
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        old = list(CKPT_DIR.glob("gated_curriculum_*_steps.zip"))
        if old:                                    # keep old checkpoints, but out of --resume's way
            keep = CKPT_DIR / f"old_{stamp}"
            keep.mkdir()
            for f in old:
                f.rename(keep / f.name)
            print(f"   (old checkpoints moved to {keep})")
        prefix = f"gated_training_{stamp}"
        csv_path = LOG_DIR / f"{prefix}_episodes.csv"

    print("[1/3] Creating gated curriculum environment...")
    print("   Altitude: 30m | virtual ground 5m")
    print(f"   Max steps: 600 ({600 / HZ:.0f} s)")
    print(f"   Wind: 0-{args.wind_strength} (AirSim gusts, as Stage 2)")
    print("   Strategy: Performance-gated progression")
    env = DummyVecEnv([lambda: Monitor(DroneFlipRecoveryEnv(args.obs_norm, target_altitude=30.0, max_steps=600,
                                                            wind_strength=args.wind_strength,
                                                            flip_prob=args.flip_prob, seed=args.seed,
                                                            debug=True))])
    base_env = env.envs[0].unwrapped
    if old_rows is not None:
        restore_from_log(old_rows, base_env)
    print("   ✅ Gated curriculum environment created!")

    if ckpt:
        print(f"\n[2/3] Resuming from checkpoint: {ckpt}")
        model = PPO.load(str(ckpt), env=env, device="cpu", tensorboard_log=TB_DIR)
        print(f"   ✅ Resumed ({model.num_timesteps:,} steps)")
    else:
        print("\n[2/3] Loading Stage 2 trained model...")
        print(f"   Model path: {args.stage2_model}")
        if not Path(args.stage2_model).exists():
            raise SystemExit(f"   ❌ Stage 2 model not found: {args.stage2_model}")
        model = PPO.load(args.stage2_model, env=env, device="cpu", tensorboard_log=TB_DIR)
        model.num_timesteps = 0
        print("   ✅ Stage 2 policy loaded!")
        print("   ✅ TensorBoard logging enabled!")
    model.learning_rate = args.lr
    model.lr_schedule = get_schedule_fn(args.lr)
    model.verbose = 1
    print(f"   📉 Learning rate: {args.lr}")

    progress = GatedCurriculumCallbackWithLogging(base_env, csv_path, prefix, old_rows, t0_s)
    callbacks = CallbackList([ActorFreeze(args.critic_warmup), progress,
                              CheckpointCallback(save_freq=50000, save_path=str(CKPT_DIR), name_prefix="gated_curriculum")])
    remaining = args.timesteps - model.num_timesteps
    if remaining <= 0:
        raise SystemExit(f"Already at {model.num_timesteps:,} steps - raise --timesteps to train longer")

    print("\n[3/3] Starting gated curriculum training...")
    print(f"   Total timesteps: {args.timesteps:,}")
    print(f"   Learning rate: {args.lr}")
    print(f"   💾 Auto-save enabled: {LEVEL_DIR}")
    print(f"   📊 TensorBoard log: {TB_DIR}")
    print("\n" + "=" * 70)
    print("🚀 GATED TRAINING STARTED")
    print("=" * 70)
    print("Watch for curriculum advancements:")
    print("  ✅ Level 0 → 1: When easy cases reach 80% recovery")
    print("     💾 Auto-saves: level_0_EASY_mastered.zip")
    print("  ✅ Level 1 → 2: When medium cases reach 70% recovery")
    print("     💾 Auto-saves: level_1_MEDIUM_mastered.zip")
    print("  🎯 Level 2:     Target 60% recovery on hard cases")
    print("     💾 Final save: gated_curriculum_policy.zip")
    print("=" * 70 + "\n")

    status = "complete"
    try:
        try:
            import rich  # noqa: F401
            import tqdm  # noqa: F401
            bar = True
        except ImportError:
            bar = False
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False,
                    tb_log_name="PPO", progress_bar=bar)
        model.save(str(MODELS / "gated_curriculum_policy"))
    except KeyboardInterrupt:
        status = "interrupted"
        p = CKPT_DIR / f"gated_curriculum_{model.num_timesteps}_steps.zip"
        model.save(str(p))
        print("\n⚠️  Training interrupted")
        print(f"💾 Model saved at interruption point: {p}")
        print("   Continue with: py .\\train_stage_3_gated_curriculum.py --resume")
    except Exception:
        status = "crashed"
        p = CKPT_DIR / f"gated_curriculum_{model.num_timesteps}_steps.zip"
        try:
            model.save(str(p))
            print(f"\n❌ Training stopped by an error (details below). Model saved at that point: {p}")
        except Exception:
            print("\n❌ Training stopped by an error (details below). Last 50k checkpoint is kept.")
        print("   Fix the cause (restart SITL if it crashed), then: py .\\train_stage_3_gated_curriculum.py --resume")
        raise
    finally:
        summary = progress.save_final_summary()
        progress.close()
        if status == "complete":
            print("\n" + "=" * 70)
            print("✅ GATED TRAINING COMPLETE!")
            print("=" * 70)
            print("\n💾 Models saved:")
            print("\n   📂 Curriculum Level Models (Auto-saved during training):")
            for lv in (0, 1):
                f = LEVEL_DIR / f"level_{lv}_{LEVEL_NAMES[lv]}_mastered.zip"
                if f.exists():
                    print(f"      ✅ Level {lv} ({LEVEL_NAMES[lv]}): {f}")
            print("\n   📂 Final Model:")
            print(f"      ✅ {MODELS / 'gated_curriculum_policy.zip'}")
            print("\n📊 Training Statistics:")
            print(f"   Total episodes: {progress.episode_count}")
            fp = summary["final_performance"]
            print(f"   Avg return: {fp['last_50_avg_reward']:.1f} (last 50)")
            print(f"   Recovery rate: {fp['last_50_recovery_rate']:.0f}% (last 50)")
            if fp["avg_recovery_time_steps"]:
                print(f"   Avg recovery time: {fp['avg_recovery_time_steps']:.0f} steps ({fp['avg_recovery_time_seconds']:.1f}s)")
            print("\n🎓 Curriculum Progression:")
            for level, d in sorted(progress.level_achievements.items()):
                print(f"   Level {level} ({LEVEL_LABELS[level]}): Reached at episode {d['reached_at_episode']} "
                      f"({d['elapsed_time'] / 3600:.1f}h)")
            print("\n✅ Next Steps:")
            print("   1. Create learning curves + tables:")
            print("      py .\\analyze_training_logs.py --output-dir .\\thesis_figures")
            print("   2. Test overall performance (and the ArduPilot baseline):")
            print("      py .\\test_stage_3_gated_curriculum.py --test-level -1 --episodes 60")
            print("      py .\\test_stage_3_gated_curriculum.py --model ardupilot --test-level -1 --episodes 60")
            print("   3. Curriculum comparison (no forgetting):")
            print("      py .\\compare_curriculum_model.py --episodes 20")
            print("=" * 70 + "\n")
        try:
            base_env.land()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-model", default="../Stage_2/runs/run1/models/warm/hover_disturbance_policy.zip",
                    help="Path to Stage 2 trained policy")
    ap.add_argument("--obs-norm", default="../Stage_1/runs/run1/models/obs_norm.npz",
                    help="Stage 1 input scaling (same as Stage 2)")
    ap.add_argument("--timesteps", type=int, default=600000, help="Total training timesteps")
    ap.add_argument("--wind-strength", type=float, default=5.0, help="Maximum wind (AirSim setting)")
    ap.add_argument("--flip-prob", type=float, default=1.0, help="Disturbance probability")
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate (AirSim Stage 3: 1e-5)")
    ap.add_argument("--critic-warmup", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint")
    ap.add_argument("--new-run", action="store_true", help="start a new training although logs exist")
    main(ap.parse_args())
