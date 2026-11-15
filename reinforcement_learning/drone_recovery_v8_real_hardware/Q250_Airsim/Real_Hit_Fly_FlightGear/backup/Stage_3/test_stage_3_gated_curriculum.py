"""
GATED CURRICULUM TESTING SCRIPT (ArduPilot SITL + AirSim)
=========================================================
Same printout and same analysis as your AirSim test_stage_3_gated_curriculum.py:
tests the gated-curriculum model at EASY / MEDIUM / HARD, recovery rate per level,
detailed intensity breakdown, episode breakdown, grade, PhD assessment.

NEW: --model ardupilot = the BASELINE: no network, ArduPilot holds still after the impact
(same environment, same impacts, same recovery rule) - the number the network must beat.
NEW: results are saved (analysis_results/test_stage3_<model>_<date>.json / .csv).
Fixed seeds: every model sees the same impacts (seed per episode = 5000 + episode number).
The curriculum is frozen during tests (AirSim's test could advance the level by itself).

Usage (in Stage_3, AirSim + SITL in GPS mode running):
    py .\\test_stage_3_gated_curriculum.py                                   (final model, HARD, 60)
    py .\\test_stage_3_gated_curriculum.py --test-level -1 --episodes 60     (all levels, 20 each)
    py .\\test_stage_3_gated_curriculum.py --model ardupilot --test-level -1 --episodes 60   (baseline)
    py .\\test_stage_3_gated_curriculum.py --test-level 2 --episodes 60 --wind 0 / 5 / 10 / 15
    py .\\test_stage_3_gated_curriculum.py --model .\\models\\stage3_checkpoints\\curriculum_levels\\level_0_EASY_mastered.zip
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np

from drone_flip_recovery_env_injector_gated import HZ, DroneFlipRecoveryEnv

LEVEL_NAMES = ["EASY (0.7-0.9)", "MEDIUM (0.9-1.1)", "HARD (1.1-1.5)"]
THRESHOLDS = [0.80, 0.70, 0.60]
SEED0 = 5000
OUT = Path("./analysis_results")


def load_policy(model):
    """Returns (act(obs) -> action, description). 'ardupilot' = baseline (hold still)."""
    if model == "ardupilot":
        return (lambda obs: np.zeros(3, dtype=np.float32)), "ArduPilot baseline (GUIDED, hold still)"
    from stable_baselines3 import PPO
    if not Path(model).exists():
        raise SystemExit(f"❌ Model not found: {model}")
    m = PPO.load(model, device="cpu")
    return (lambda obs: m.predict(obs, deterministic=True)[0]), model


def run_episode(env, act, level, seed):
    env.curriculum_level = level
    obs, _ = env.reset(options={"seed": seed})
    step, info = 0, {}
    while True:
        obs, r, term, trunc, info = env.step(act(obs))
        step += 1
        if term or trunc:
            return step, info


def test_gated_curriculum(args):
    act, desc = load_policy(args.model)
    print("\n" + "=" * 70)
    print("🧪 TESTING GATED CURRICULUM MODEL")
    print("=" * 70)
    print(f"Model: {desc}")
    print(f"Episodes: {args.episodes}")
    print(f"Wind Strength: {args.wind} (AirSim setting)")
    wind_desc = ("No wind (calm conditions)" if args.wind == 0 else "Light breeze" if args.wind <= 3 else
                 "Moderate wind" if args.wind <= 7 else "Strong wind" if args.wind <= 12 else "Very strong wind")
    print(f"Wind Condition: {wind_desc}")
    meta = Path(args.model).with_name(Path(args.model).stem + "_metadata.json") if args.model != "ardupilot" else None
    if meta and meta.exists():
        md = json.load(open(meta))
        print("\n📋 Model Info (from metadata):")
        print(f"   Level: {md.get('level')} - {md.get('level_name')}")
        print(f"   Recovery rate at save: {md.get('recovery_rate', 0) * 100:.1f}%")
        print(f"   Episode: {md.get('episode')}")
        print(f"   Training time: {md.get('timestamp', 0) / 3600:.1f}h")
    levels = [0, 1, 2] if args.test_level == -1 else [args.test_level]
    per_level = args.episodes // len(levels)
    print(f"Testing: {'ALL intensity levels' if args.test_level == -1 else LEVEL_NAMES[args.test_level] + ' only'}")
    print("=" * 70 + "\n")
    print("[1/3] Creating test environment...")
    env = DroneFlipRecoveryEnv(args.obs_norm, target_altitude=30.0, max_steps=600, wind_strength=args.wind,
                               flip_prob=1.0, seed=0, debug=True, curriculum_frozen=True)
    print("[2/3] Input scaling: Stage 1 obs_norm.npz (fixed)")
    print("[3/3] Model loaded")
    print("   ✅ Model loaded successfully!\n")
    print(f"Testing {per_level} episodes per level\n")
    print("=" * 70)
    print("🚀 STARTING TEST EPISODES")
    print("=" * 70 + "\n")

    rows, n = [], 0
    try:
        for lv in levels:
            print(f"\n{'=' * 70}")
            print(f"📊 TESTING LEVEL {lv}: {LEVEL_NAMES[lv]}")
            print(f"{'=' * 70}\n")
            for k in range(per_level):
                n += 1
                print(f"Episode {n} (Level {lv}, #{k + 1}):")
                steps, info = run_episode(env, act, lv, SEED0 + n)
                success = info["reason"] == "timeout"
                rows.append({"episode": n, "level": lv, "seed": SEED0 + n, "steps": steps, "reason": info["reason"],
                             "success": success, "distance": info["final_distance"],
                             "tumble_initiated": info["tumble_initiated"], "tumble_recovered": info["tumble_recovered"],
                             "recovery_steps": info["recovery_steps"], "recovery_time_s": info["recovery_time_s"],
                             "intensity": info["disturbance_intensity"], "type": info["disturbance_type"],
                             "turns": info["impact_turns"], "min_altitude": info["min_altitude"],
                             "max_angular_velocity": info["max_angular_velocity"]})
                print(f"   {'✅' if success else '💥'} Steps: {steps} | Dist: {info['final_distance']:.2f}m | "
                      f"Reason: {info['reason']}")
                if info["tumble_initiated"]:
                    st = "✅ Recovered" if info["tumble_recovered"] else "❌ Crashed"
                    print(f"      Disturbance: {info['disturbance_type']} | Intensity {info['disturbance_intensity']:.2f}x "
                          f"| {info['impact_turns']:.1f} turns | {st} | lowest {info['min_altitude']:.1f} m")
                    if info["tumble_recovered"]:
                        print(f"      Recovery time: {info['recovery_steps']} steps ({info['recovery_time_s']:.1f}s)")
                print()
    except KeyboardInterrupt:
        print("\nStopped by user - analysing what we have")
    finally:
        report(rows, levels, desc, args)
        env.land()
        env.close()


def report(rows, levels, desc, args):
    if not rows:
        return
    succ = np.array([r["success"] for r in rows])
    ini = np.array([r["tumble_initiated"] for r in rows])
    rec = np.array([r["tumble_recovered"] for r in rows])
    ints = np.array([r["intensity"] for r in rows])
    lvls = np.array([r["level"] for r in rows])
    rt = [r["recovery_steps"] for r in rows if r["tumble_recovered"]]
    rate = rec[ini].mean() * 100 if ini.any() else 0.0
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    print("\n📈 Overall Performance:")
    print(f"   Success Rate: {succ.mean() * 100:.0f}% ({succ.sum()}/{len(rows)} episodes)")
    if succ.any():
        print(f"   Average Distance: {np.mean([r['distance'] for r in rows if r['success']]):.2f}m (successful episodes)")
    print(f"   Average Episode Length: {np.mean([r['steps'] for r in rows]):.1f} steps")
    print("\n🔄 Overall Recovery:")
    print(f"   Episodes with disturbance: {ini.sum()}/{len(rows)}")
    print(f"   Successful recoveries: {rec.sum()}/{ini.sum()}")
    print(f"   Overall Recovery Rate: {rate:.0f}%")
    if rt:
        print(f"   Avg Recovery Time: {np.mean(rt):.0f} steps ({np.mean(rt) / HZ:.1f}s)")
    print(f"   Avg Lowest Height: {np.mean([r['min_altitude'] for r in rows]):.1f} m (virtual ground 5 m)")

    print(f"\n{'=' * 70}")
    print("🎯 PERFORMANCE BY CURRICULUM LEVEL (PhD Analysis)")
    print(f"{'=' * 70}")
    by_level = {}
    for lv in levels:
        m = lvls == lv
        if m.any():
            lr = rec[m].mean() * 100
            thr = THRESHOLDS[lv] * 100
            lrt = [r["recovery_time_s"] for r in rows if r["level"] == lv and r["tumble_recovered"]]
            by_level[lv] = {"episodes": int(m.sum()), "recovered": int(rec[m].sum()), "recovery_rate": float(lr),
                            "avg_intensity": float(ints[m].mean()),
                            "avg_recovery_time_s": float(np.mean(lrt)) if lrt else None,
                            "avg_min_altitude_m": float(np.mean([r["min_altitude"] for r in rows if r["level"] == lv]))}
            print(f"\n   Level {lv} - {LEVEL_NAMES[lv]}:")
            print(f"      Recovery Rate: {lr:.1f}% ({rec[m].sum()}/{m.sum()}) "
                  f"{'✅ MASTERED' if lr >= thr else '⚠️  BELOW TARGET'}")
            print(f"      Target: {thr:.0f}%")
            print(f"      Avg Intensity: {ints[m].mean():.2f}x")
            if lrt:
                print(f"      Avg Recovery Time: {np.mean(lrt):.2f}s")
            if lr < thr:
                print(f"      💡 Need {thr - lr:.0f}% more to reach target")
    print(f"\n{'=' * 70}")

    print("\n🔬 DETAILED INTENSITY BREAKDOWN:")
    print(f"{'=' * 70}")
    for lo, hi, name in ((0.7, 0.9, "0.7-0.9x (EASY):  "), (0.9, 1.1, "0.9-1.1x (MEDIUM):"),
                         (1.1, 1.3, "1.1-1.3x (HARD):  "), (1.3, 9.9, "1.3-1.5x (EXTREME):")):
        m = (ints >= lo) & (ints < hi) & ini
        if m.any():
            print(f"   {name} {rec[m].mean() * 100:.1f}% ({rec[m].sum()}/{m.sum()})")
    print(f"{'=' * 70}")
    print("\n🔬 BY DISTURBANCE TYPE:")
    for t in ("flip", "spin", "bird_attack"):
        m = np.array([r["type"] == t for r in rows]) & ini
        if m.any():
            print(f"   {t:<12} {rec[m].mean() * 100:.1f}% ({rec[m].sum()}/{m.sum()})")

    print("\n📊 Episode Breakdown:")
    reasons = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    for k, v in reasons.items():
        print(f"   {k}: {v} episodes")

    print("\n🎯 Performance Grade:")
    grade = ("A+ (OUTSTANDING - PhD COMPLETE!)" if rate >= 75 else "A (EXCELLENT - PhD TARGET MET!)" if rate >= 65
             else "B+ (VERY GOOD - Close to target)" if rate >= 55 else "B (GOOD - More training needed)"
             if rate >= 45 else "C (NEEDS IMPROVEMENT - Continue training)")
    print(f"   Overall Performance: {grade}")
    print("\n🎓 PhD GATED CURRICULUM ASSESSMENT:")
    all_ok = True
    for lv, s in by_level.items():
        thr = THRESHOLDS[lv] * 100
        ok = s["recovery_rate"] >= thr
        all_ok &= ok
        print(f"   {'✅' if ok else '❌'} Level {lv} ({LEVEL_NAMES[lv]}): "
              f"{'MASTERED' if ok else 'Below target'} ({s['recovery_rate']:.0f}% {'≥' if ok else '<'} {thr:.0f}%)")
    if all_ok and rate >= 65:
        print("\n   🎉 ALL LEVELS MASTERED!\n   ✅ PhD requirements met\n   ✅ Performance-gated curriculum validated")
    elif all_ok:
        print("\n   ✅ All individual levels mastered\n   ⚠️  Overall rate slightly below 65% target")
    else:
        print("\n   ⚠️  Some levels below mastery threshold")
    print("\n" + "=" * 70 + "\n")

    OUT.mkdir(exist_ok=True)
    tag = "ardupilot" if args.model == "ardupilot" else Path(args.model).stem
    base = OUT / f"test_stage3_{tag}_wind{args.wind:g}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    with open(f"{base}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({"model": desc, "settings": vars(args), "overall_recovery_rate": float(rate),
               "success_rate": float(succ.mean() * 100),
               "avg_recovery_time_s": float(np.mean(rt) / HZ) if rt else None,
               "by_level": {str(k): v for k, v in by_level.items()}, "episodes": rows},
              open(f"{base}.json", "w"), indent=2, default=str)
    print(f"💾 Saved: {base}.csv / .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test gated curriculum models (ArduPilot)")
    ap.add_argument("--model", default="./models/stage3_checkpoints/gated_curriculum_policy.zip",
                    help="Path to trained model, or 'ardupilot' for the baseline")
    ap.add_argument("--obs-norm", default="../Stage_1/runs/run1/models/obs_norm.npz")
    ap.add_argument("--episodes", type=int, default=60, help="Number of test episodes (split over levels if -1)")
    ap.add_argument("--test-level", type=int, default=2, help="-1=all, 0=easy, 1=medium, 2=hard")
    ap.add_argument("--wind", type=float, default=0.0, help="Wind strength (AirSim setting, 0 = calm)")
    test_gated_curriculum(ap.parse_args())
