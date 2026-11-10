r"""
STAGE 3 PPO - PERFORMANCE-GATED CURRICULUM TRAINING  (step 3c, ArduPilot SITL + AirSim)
======================================================================================

1. HOW TO RUN  (PowerShell in Stage_3, AirSim + SITL in GPS mode running)
------------------------------------------------------------------------
    Needs models\stage3_ppo_init.zip + models\obs_norm.npz (from train_imitation.py).

    First a ~5-10 min smoke test (one PPO update, saves everything, then stops by itself):

    py .\train_stage_3_gated_curriculum.py --new-run --timesteps 2048

    The real training (~7 h for 200,000 steps - run it overnight). It starts a NEW log; the smoke
    test's files and any old Stage 3 models are moved to models\stage3_checkpoints\old_<date>\ :

    py .\train_stage_3_gated_curriculum.py --new-run

    Stopped (Ctrl+C), PC restarted or SITL crashed? Restart SITL, then continue exactly where it stopped:

    py .\train_stage_3_gated_curriculum.py --resume

    Train longer after it finished (e.g. to 300,000 steps):

    py .\train_stage_3_gated_curriculum.py --resume --timesteps 300000

    ROUND 2 (faster recovery) - start from the round 1 model, new log, overnight (~7 h):
    first copy the round 1 model (a new run moves the old models to old_<date>):

    copy models\stage3_checkpoints\gated_curriculum_policy.zip models\round1_final.zip

    py .\train_stage_3_gated_curriculum.py --new-run --init-model models\round1_final.zip --lr 5e-5 --clip 0.2 --time-weight 0.15 --flip-share 0.6

    Watch the curves while it trains (second PowerShell, then open http://localhost:6006):

    tensorboard --logdir logs\gated_curriculum


2. EXPECTED RESULT  (estimates - your first run gives the real numbers)
------------------------------------------------------------------------
    Speed            ~350 episodes per hour (fly back + hit + recovery ~10 s), ~30,000 steps per hour
    First 20,000 st  "critic warm-up": the network is frozen = BC behaviour (recovery ~95-100 %)
    EASY -> MEDIUM   after ~50-80 episodes (needs 80 %; BC already recovers ~100 %)
    MEDIUM -> HARD   after ~100-160 episodes (needs 70 %)
    HARD             recovery >= 60 % (target; BC is already ~100 %), lowest height should RISE over time
    Every 10 episodes a block like:
        📊 EPISODE 120 | 0.4h elapsed
           🎓 Curriculum Level: 2 - HARD (1.1-1.5)
           Recovery Rate: 100% | Avg lowest: 17.2 m | Avg Recovery Time: 38 steps (0.8s)

    EARLY CHECK - printed once at --early-check steps (default 60,000, ~2 h) and saved to
    logs\stage3\gated_training_<date>_early_check.json. It compares PPO (after warm-up) with the
    frozen-network episodes (= BC) on the SAME level:
    Result                                           Meaning                    Action
    lowest height up, recovery % same                PPO is improving           let it run
    about the same as BC                             still learning slowly      let it run, check again at 4 h
    recovery % down > 10 points, or lowest down >2 m PPO is forgetting BC       Ctrl+C, send me the CSV


3. OUTPUT FILES  (same names as your AirSim Stage 3 - analyze_training_logs.py reads them)
------------------------------------------------------------------------
    logs\stage3\gated_training_<date>_episodes.csv      one line per episode (written live, never lost)
    logs\stage3\gated_training_<date>_summary.json      final numbers (per level, lowest height, time)
    logs\stage3\gated_training_<date>_curriculum.json   level per episode, recoveries, intensities
    logs\stage3\gated_training_<date>_early_check.json  the early check (section 2)
    logs\gated_curriculum\gated_training_<date>_0\     TensorBoard (one folder per run; --resume continues it)
    models\stage3_checkpoints\curriculum_levels\level_0_EASY_mastered.zip  (+ _metadata.json)
    models\stage3_checkpoints\curriculum_levels\level_1_MEDIUM_mastered.zip (+ _metadata.json)
    models\stage3_checkpoints\gated_checkpoints\gated_curriculum_<steps>_steps.zip   every 20,000 steps
    models\stage3_checkpoints\gated_curriculum_policy.zip                             FINAL model
    models\stage3_checkpoints\gated_curriculum_best.zip (+ _metadata.json)            BEST model: best flip lowest
        height - recovery time over the last 100 HARD episodes (>= 95 % recovered), checked every 10 episodes


4. WHAT THE SCRIPT DOES
------------------------------------------------------------------------
    Step 1  load the BC network (models\stage3_ppo_init.zip) as PPO's starting point
    Step 2  set PPO's exploration small (std 0.1 rad / 0.1 x hover) so it starts AS the BC network
    Step 3  critic warm-up: first 20,000 steps the network is frozen, only the value net learns
    Step 4  PPO learns; every episode = one hit + one recovery (see the environment file)
    Step 5  gated curriculum: EASY -> MEDIUM -> HARD only when the level is mastered
    Step 6  logs every episode, saves checkpoints, level models, the final model and the summaries


5. PPO SETTINGS  (--option, default)
------------------------------------------------------------------------
    --timesteps 200000   --lr 3e-5   --critic-warmup 20000   --early-check 60000   --seed 0
    n_steps 2048 | batch 64 | epochs 10 | gamma 0.99 | GAE 0.95 | clip 0.1 | target_kl 0.02 | std 0.1
    Randomisation (training only):  --obs-noise 1.0   --thrust-rand 0.05   --wind-strength 3.0
    Network 13-256-256-128-3 ReLU (same as BC), input scaling models\obs_norm.npz (fixed)


6. WHAT IS COUNTED  (for the curriculum and the thesis)
------------------------------------------------------------------------
    RECOVERED   PPO reached CALM (rate < 1.2 rad/s, tilt < 30 deg, falling < 1 m/s) above 5 m, within 8 s
    FAILED      CRASH (virtual ground 5 m) or TIMEOUT
    NOT COUNTED episodes cut by a PPO update ('update_pause'): not in the CSV, not in the curriculum
    The final thesis comparison (PPO vs BC vs attitude-rule vs ArduPilot, same hits) is done after
    training with test_stage_3_gated_curriculum.py - see "Next steps" at the end of training.


7. SAFETY / CRASH RECOVERY OF THE TRAINING ITSELF
------------------------------------------------------------------------
    - Ctrl+C or any error -> checkpoint saved at that exact step, logs closed cleanly
    - --resume continues: model, step count, curriculum level (rebuilt from the CSV), same log files
    - episodes logged after the checkpoint are removed from the CSV on --resume (they are re-flown)
    - a new run never overwrites old logs (needs --new-run; old checkpoints are moved to old_<date>)
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
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv

from drone_flip_recovery_env_injector_gated import BRAKE, FLOOR, HZ, LEVEL_NAMES, DroneFlipRecoveryEnv
from train_imitation import HIDDEN

BEST_WINDOW = 100          # best model: judged on the last 100 HARD episodes after warm-up
INIT_STD = 0.1             # exploration noise at the start (rad for roll/pitch, x hover for thrust)

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
            "impact_peak_rate", "max_angular_velocity", "final_distance",
            # new design (attitude + thrust recovery)
            "height_loss_m", "alt_end_impact_m", "impact_cut", "rescued", "actor_frozen", "thrust_k"]


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
        self.lowest, self.frozen_flags, self.types, self.rec_time_ep = [], [], [], []
        self.best_score, self.best_info = None, None
        self.freeze = None                                    # ActorFreeze callback (set in main)
        self.settings = {}                                    # command-line settings (set in main)
        self.early_steps, self.early_done = None, False
        new = old_rows is None
        for r in old_rows or []:
            self._remember(float(r["episode_reward"]), int(float(r["episode_length"])), int(r["curriculum_level"]),
                           int(r["tumble_initiated"]), int(r["tumble_recovered"]), int(float(r["recovery_steps"])),
                           float(r["disturbance_intensity"]), float(r["elapsed_time_s"]), float(r["timestamp"]),
                           float(r.get("min_altitude") or 0), int(r.get("actor_frozen") or 0), r["disturbance_type"])
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

    def _remember(self, ret, length, level, initiated, recovered, rec_steps, intensity, elapsed, stamp,
                  lowest=0.0, frozen=0, kind=""):
        self.lowest.append(lowest)
        self.rec_time_ep.append(rec_steps / HZ if (initiated and recovered) else None)
        self.frozen_flags.append(frozen)
        self.types.append(kind)
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
            frozen = int(bool(self.freeze and self.freeze.frozen))
            self._remember(info["episode_return"], info["episode_steps"], level, ini, rec,
                           info["recovery_steps"], info["disturbance_intensity"], elapsed, now,
                           info["min_altitude"], frozen, info["disturbance_type"])
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
                round(info["final_distance"], 2),
                round(info["height_loss_m"], 2), round(info["alt_end_impact_m"], 2), int(info["impact_cut"]),
                int(info["rescued"]), frozen, round(info["thrust_k"], 3)])
            self.csv_file.flush()                       # never lose data on a crash
            self.logger.record("stage3/recovery_rate_last50_pct", rr50)
            self.logger.record("stage3/return_last10", r10)
            self.logger.record("stage3/curriculum_level", level)
            self.logger.record("stage3/lowest_altitude_m", info["min_altitude"])
            if self.early_steps and not self.early_done and self.num_timesteps >= self.early_steps:
                self.early_check()
            if self.base_env.level_advanced:
                self._save_advancement_model(self.base_env.advancement_info)
                self.base_env.level_advanced = False
            if level == 2 and not frozen and self.episode_count % 10 == 0:
                self._check_best()
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
        print(f"      Avg Lowest Height: {np.mean(self.lowest[-10:]):.1f} m (start 30 m, virtual ground {FLOOR:.0f} m)")
        if self.freeze and self.freeze.frozen:
            print("      (critic warm-up: network frozen = BC behaviour)")
        target = [80, 70, 60][lvl]
        print(f"   📈 Progress: {rate:.0f}% / {target}% " + ("(PhD target)" if lvl == 2 else "(need to advance)"))
        print(f"{'=' * 70}\n")

    def _check_best(self):
        """BEST MODEL: last 100 HARD episodes after warm-up. Needs >= 95 % recovered; score = flip avg
        lowest height (m) - flip avg recovery time (s). Higher = better. Saved as gated_curriculum_best.zip."""
        idx = [i for i in range(len(self.curriculum_levels))
               if self.curriculum_levels[i] == 2 and not self.frozen_flags[i]][-BEST_WINDOW:]
        if len(idx) < BEST_WINDOW:
            return
        rate = float(np.mean([self.recovery_successes[i] for i in idx]))
        flips = [i for i in idx if self.types[i] == "flip"]
        times = [self.rec_time_ep[i] for i in flips if self.rec_time_ep[i] is not None]
        if rate < 0.95 or len(flips) < BEST_WINDOW // 5 or not times:
            return
        low, t = float(np.mean([self.lowest[i] for i in flips])), float(np.mean(times))
        score = low - t
        if self.best_score is not None and score <= self.best_score:
            return
        self.best_score = score
        self.best_info = {"timesteps": int(self.num_timesteps), "episode": self.episode_count, "score": score,
                          "flip_avg_lowest_m": low, "flip_avg_recovery_s": t, "hard_recovery_rate": rate * 100,
                          "window": f"last {BEST_WINDOW} HARD episodes after warm-up", "flips_in_window": len(flips)}
        self.model.save(str(MODELS / "gated_curriculum_best"))
        with open(MODELS / "gated_curriculum_best_metadata.json", "w") as f:
            json.dump(self.best_info, f, indent=2)
        print(f"   🏆 New best model (step {self.num_timesteps:,}): flip lowest {low:.1f} m | flip recovery "
              f"{t:.2f} s | HARD recovered {rate * 100:.0f}% -> {MODELS / 'gated_curriculum_best.zip'}")

    def early_check(self):
        """Once, at --early-check steps: PPO (after warm-up) vs frozen network (= BC) on each level."""
        self.early_done = True
        n = len(self.episode_rewards)
        res = {"timesteps": int(self.num_timesteps), "episodes": n, "levels": {}}
        print("\n" + "=" * 70)
        print(f"🔎 EARLY CHECK at {self.num_timesteps:,} steps ({n} episodes)")
        print("=" * 70)
        print(f"   {'level':<8} {'who':<5} {'episodes':>8} {'recovered':>10} {'avg lowest':>11}")
        verdicts = []
        for lv in range(3):
            row = {}
            for who, fz in (("BC", 1), ("PPO", 0)):
                idx = [i for i in range(n) if self.curriculum_levels[i] == lv and self.frozen_flags[i] == fz]
                if fz == 0:
                    idx = idx[-50:]
                if not idx:
                    continue
                rr = float(np.mean([self.recovery_successes[i] for i in idx]) * 100)
                lo = float(np.mean([self.lowest[i] for i in idx]))
                row[who] = {"episodes": len(idx), "recovered_pct": rr, "avg_lowest_m": lo}
                print(f"   {LEVEL_NAMES[lv]:<8} {who:<5} {len(idx):>8} {rr:>9.0f}% {lo:>10.1f} m")
            if "BC" in row and "PPO" in row:
                d_rr = row["PPO"]["recovered_pct"] - row["BC"]["recovered_pct"]
                d_lo = row["PPO"]["avg_lowest_m"] - row["BC"]["avg_lowest_m"]
                v = ("PPO is forgetting BC - Ctrl+C and send me the CSV" if d_rr < -10 or d_lo < -2 else
                     "PPO is improving - let it run" if d_lo > 0.3 and d_rr >= -5 else
                     "about the same as BC - let it run, check again later")
                row["verdict"] = v
                verdicts.append(f"{LEVEL_NAMES[lv]}: {v}")
            res["levels"][LEVEL_NAMES[lv]] = row
        if not verdicts:
            print("   (no level has both BC and PPO episodes yet - compare in the CSV later)")
        for v in verdicts:
            print(f"   ➜ {v}")
        print("   (BC episodes = the first 20,000 steps when the network was frozen; PPO = last 50 after it)")
        print("=" * 70 + "\n")
        with open(LOG_DIR / f"{self.log_prefix}_early_check.json", "w") as f:
            json.dump(res, f, indent=2)

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
                "obs_norm": "models/obs_norm.npz (BC scaling, fixed, no VecNormalize)"}
        with open(LEVEL_DIR / f"{name}_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"   ✅ Model saved: {path}.zip")
        print(f"   ✅ Metadata saved: {LEVEL_DIR / f'{name}_metadata.json'}")
        print("=" * 70 + "\n")

    def save_final_summary(self):
        hours = (time.time() - self.start_time) / 3600
        last = lambda v, n: v[-n:] if len(v) >= n else v
        summary = {
            "settings": self.settings,
            "best_model": self.best_info,
            "training_info": {"total_episodes": self.episode_count, "total_time_seconds": hours * 3600,
                              "total_time_hours": hours, "episodes_per_hour": self.episode_count / max(hours, 1e-9),
                              "episodes_cut_by_ppo_update": self.update_pauses, "end_time": time.time(),
                              "virtual_ground_m": FLOOR, "crash_rule_brake_mss": BRAKE},
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
                ppo = [i for i in eps if not self.frozen_flags[i]]
                summary["statistics_by_level"][f"level_{level}_{LEVEL_NAMES[level]}"] = {
                    "episodes_trained": len(eps), "recovery_rate": float(np.mean(rec) * 100) if rec else 0,
                    "avg_reward": float(np.mean([self.episode_rewards[i] for i in eps])),
                    "avg_intensity": float(np.mean(ints)) if ints else 0,
                    "avg_lowest_m": float(np.mean([self.lowest[i] for i in eps])),
                    "last_50_ppo_avg_lowest_m": float(np.mean([self.lowest[i] for i in ppo[-50:]])) if ppo else None,
                    "last_50_ppo_recovery_rate": float(np.mean([self.recovery_successes[i] for i in ppo[-50:]]) * 100)
                    if ppo else None,
                    "by_type_avg_lowest_m": {k: float(np.mean([self.lowest[i] for i in eps if self.types[i] == k]))
                                             for k in sorted(set(self.types[i] for i in eps))}}
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
        old_models = list(LEVEL_DIR.glob("level_*")) + list(MODELS.glob("gated_curriculum_policy*")) + list(MODELS.glob("gated_curriculum_best*"))
        if old or old_models:                      # keep old files, but out of --resume's / the tests' way
            keep = MODELS / f"old_{stamp}"
            keep.mkdir()
            for f in old + old_models:
                f.rename(keep / f.name)
            print(f"   (old checkpoints / level models / final model moved to {keep})")
        prefix = f"gated_training_{stamp}"
        csv_path = LOG_DIR / f"{prefix}_episodes.csv"

    for f in (args.init_model, args.obs_norm):
        if not Path(f).exists():
            raise SystemExit(f"   ❌ {f} not found - run train_imitation.py first (in Stage_3)")
    print("[1/3] Creating gated curriculum environment...")
    print("   One episode = one hit + one recovery | start 30 m | virtual ground 5 m | max 8 s")
    print(f"   Randomisation: obs noise x{args.obs_noise} | thrust +-{args.thrust_rand * 100:.0f}% | "
          f"wind 0-{args.wind_strength} m/s")
    print("   Strategy: Performance-gated progression")
    env = DummyVecEnv([lambda: Monitor(DroneFlipRecoveryEnv(args.obs_norm, wind_strength=args.wind_strength,
                                                            time_weight=args.time_weight, flip_share=args.flip_share,
                                                            obs_noise=args.obs_noise, thrust_rand=args.thrust_rand,
                                                            seed=args.seed, debug=True, connect=args.conn))])
    base_env = env.envs[0].unwrapped
    if old_rows is not None:
        restore_from_log(old_rows, base_env)
    print("   ✅ Gated curriculum environment created!")

    if ckpt:
        print(f"\n[2/3] Resuming from checkpoint: {ckpt}")
        model = PPO.load(str(ckpt), env=env, device="cpu", tensorboard_log=TB_DIR)
        print(f"   ✅ Resumed ({model.num_timesteps:,} steps)")
    else:
        print("\n[2/3] Loading the BC network as PPO's starting point...")
        print(f"   Model path: {args.init_model}")
        model = PPO("MlpPolicy", env, learning_rate=args.lr, n_steps=2048, batch_size=64, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, clip_range=args.clip, ent_coef=0.0, target_kl=0.02, seed=args.seed,
                    policy_kwargs=dict(net_arch=dict(pi=HIDDEN, vf=HIDDEN), activation_fn=nn.ReLU),
                    device="cpu", tensorboard_log=TB_DIR, verbose=1)
        init = PPO.load(args.init_model, device="cpu").policy.state_dict()
        actor = {k: v for k, v in init.items()
                 if k.startswith("mlp_extractor.policy_net") or k.startswith("action_net")}
        res = model.policy.load_state_dict(actor, strict=False)
        if res.unexpected_keys or any(k.startswith(("mlp_extractor.policy_net", "action_net"))
                                      for k in res.missing_keys):
            raise SystemExit(f"   ❌ BC network did not load: {res}")
        with torch.no_grad():
            model.policy.log_std.fill_(float(np.log(INIT_STD)))
        print(f"   ✅ BC actor loaded ({len(actor)} tensors) | exploration std {INIT_STD}")
        print("   ✅ TensorBoard logging enabled!")
    model.learning_rate = args.lr
    model.lr_schedule = get_schedule_fn(args.lr)
    model.clip_range = get_schedule_fn(args.clip)
    model.verbose = 1
    print(f"   📉 Learning rate: {args.lr} | clip {args.clip} | reward time weight {args.time_weight} | "
          f"flip share {args.flip_share if args.flip_share is not None else 0.45}")

    progress = GatedCurriculumCallbackWithLogging(base_env, csv_path, prefix, old_rows, t0_s)
    freeze = ActorFreeze(args.critic_warmup)
    progress.freeze = freeze
    best_meta = MODELS / "gated_curriculum_best_metadata.json"
    if ckpt and best_meta.exists():                        # --resume: keep the best score found so far
        progress.best_info = json.load(open(best_meta))
        progress.best_score = progress.best_info["score"]
        print(f"   ✅ Best model so far: step {progress.best_info['timesteps']:,} (score {progress.best_score:.2f})")
    progress.settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    progress.early_steps = args.early_check if model.num_timesteps < args.early_check else None
    callbacks = CallbackList([freeze, progress,
                              CheckpointCallback(save_freq=20000, save_path=str(CKPT_DIR), name_prefix="gated_curriculum")])
    remaining = args.timesteps - model.num_timesteps
    if remaining <= 0:
        raise SystemExit(f"Already at {model.num_timesteps:,} steps - raise --timesteps to train longer")

    print("\n[3/3] Starting gated curriculum training...")
    print(f"   Total timesteps: {args.timesteps:,}")
    print(f"   Learning rate: {args.lr}")
    print(f"   💾 Auto-save enabled: {LEVEL_DIR}")
    print(f"   📊 TensorBoard log: {Path(TB_DIR) / (prefix + '_0')}  (one folder per run, same name as the CSV)")
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
                    tb_log_name=prefix, progress_bar=bar)     # own TensorBoard folder per run; --resume continues it
        model.save(str(MODELS / "gated_curriculum_policy"))
        model.save(str(CKPT_DIR / f"gated_curriculum_{model.num_timesteps}_steps"))   # so --resume can extend it
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
            print("   2. Final test - PPO vs BC vs attitude-rule vs spin-stop-fast vs ArduPilot, same hits:")
            print("      py .\\test_stage_3_gated_curriculum.py")
            print("   3. Curriculum comparison (no forgetting):")
            print("      py .\\compare_curriculum_model.py")
            print("=" * 70 + "\n")
        try:
            base_env.land()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-model", default="models/stage3_ppo_init.zip", help="BC network packed for PPO")
    ap.add_argument("--obs-norm", default="models/obs_norm.npz", help="BC input scaling (fixed)")
    ap.add_argument("--timesteps", type=int, default=200000, help="Total training timesteps")
    ap.add_argument("--wind-strength", type=float, default=3.0, help="Max wind gust m/s (AirSim), 0 = off")
    ap.add_argument("--obs-noise", type=float, default=1.0, help="Sensor noise scale, 0 = off")
    ap.add_argument("--thrust-rand", type=float, default=0.05, help="Thrust error +- fraction, 0 = off")
    ap.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    ap.add_argument("--clip", type=float, default=0.1, help="PPO clip range (new runs)")
    ap.add_argument("--time-weight", type=float, default=0.05, help="Reward time penalty per step")
    ap.add_argument("--flip-share", type=float, default=None, help="Share of flip hits, e.g. 0.6 (default 0.45)")
    ap.add_argument("--critic-warmup", type=int, default=20000)
    ap.add_argument("--early-check", type=int, default=60000, help="Early check at this many steps (~2 h)")
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint")
    ap.add_argument("--new-run", action="store_true", help="start a new training although logs exist")
    main(ap.parse_args())
