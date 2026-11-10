"""
CURRICULUM LEVEL COMPARISON - PhD Analysis (ArduPilot SITL + AirSim)
====================================================================
Same as your AirSim compare_curriculum_model.py: tests each curriculum-level model
across ALL intensity ranges -> Table 5.3 "No Catastrophic Forgetting".

Differences: one environment for all models (one ArduPilot connection), the curriculum is
frozen during the test, identical seeds for every model, the ArduPilot baseline is added
as a row, and no VecNormalize (fixed Stage 1 scaling).

Usage (in Stage_3, AirSim + SITL in GPS mode running):
    py .\\compare_curriculum_model.py --episodes 20
Saves analysis_results/curriculum_comparison_results.json and table_5_3_curriculum_comparison.tex
"""

import argparse
import json
from pathlib import Path

import numpy as np

from drone_flip_recovery_env_injector_gated import HZ, DroneFlipRecoveryEnv
from test_stage_3_gated_curriculum import load_policy, run_episode

LEVELS = Path("./models/stage3_checkpoints/curriculum_levels/")
MODELS = {
    "ArduPilot baseline": {"model": "ardupilot", "trained_on": "-"},
    "Level 0 (EASY)": {"model": str(LEVELS / "level_0_EASY_mastered.zip"), "trained_on": "EASY (0.7-0.9×)"},
    "Level 1 (MEDIUM)": {"model": str(LEVELS / "level_1_MEDIUM_mastered.zip"), "trained_on": "MEDIUM (0.9-1.1×)"},
    "Final Model": {"model": "./models/stage3_checkpoints/gated_curriculum_policy.zip", "trained_on": "HARD (1.1-1.5×)"},
}
INTENSITIES = {"EASY": 0, "MEDIUM": 1, "HARD": 2}
SEED0 = 7000


def main(args):
    print("\n" + "=" * 80)
    print("🎓 CURRICULUM LEVEL COMPARISON - PhD Analysis")
    print("=" * 80)
    print("Testing each curriculum level model across ALL intensity ranges")
    print("This demonstrates:")
    print("  1. Progressive improvement through curriculum learning")
    print("  2. No catastrophic forgetting (still works on easier cases)")
    print("  3. Generalization capabilities")
    print("=" * 80 + "\n")
    env = DroneFlipRecoveryEnv(args.obs_norm, target_altitude=30.0, max_steps=600, wind_strength=0.0,
                               flip_prob=1.0, seed=0, debug=False, curriculum_frozen=True)
    results = {}
    try:
        for name, m in MODELS.items():
            if m["model"] != "ardupilot" and not Path(m["model"]).exists():
                print(f"⚠️  {name} not found: {m['model']}")
                continue
            act, desc = load_policy(m["model"])
            print(f"\n{'=' * 80}\n📊 Testing: {name}\n   Trained on: {m['trained_on']}\n   Model: {desc}\n{'=' * 80}\n")
            results[name] = {}
            for iname, lv in INTENSITIES.items():
                rec, times, ints, lows = 0, [], [], []
                print(f"   Testing at {iname}...")
                for ep in range(args.episodes):
                    _, info = run_episode(env, act, lv, SEED0 + 100 * lv + ep)   # same impacts for every model
                    ints.append(info["disturbance_intensity"])
                    lows.append(info["min_altitude"])
                    if info["tumble_recovered"]:
                        rec += 1
                        times.append(info["recovery_steps"])
                    if (ep + 1) % 5 == 0:
                        print(f"      Episode {ep + 1}/{args.episodes}: {rec}/{ep + 1} recovered ({rec / (ep + 1) * 100:.0f}%)")
                results[name][iname] = {"recovery_rate": 100.0 * rec / args.episodes, "recoveries": rec,
                                        "attempts": args.episodes,
                                        "avg_recovery_time": float(np.mean(times)) if times else 0.0,
                                        "avg_recovery_time_s": float(np.mean(times) / HZ) if times else 0.0,
                                        "avg_intensity": float(np.mean(ints)), "std_intensity": float(np.std(ints)),
                                        "avg_min_altitude_m": float(np.mean(lows))}
                print(f"   ✅ Complete: {100.0 * rec / args.episodes:.1f}% recovery ({rec}/{args.episodes})\n")
    except KeyboardInterrupt:
        print("\nStopped by user - saving what we have")
    finally:
        env.land()
        env.close()

    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE RESULTS - PhD Table 5.3")
    print("=" * 80)
    print("Cross-Level Performance Evaluation: No Catastrophic Forgetting")
    print("-" * 80)
    print(f"\n{'Model':<22} {'Trained On':<20} {'Test Intensity':<15} {'Recovery Rate':<18} {'Avg Time (s)':<12}")
    print("-" * 80)
    for name, res in results.items():
        for i, (iname, r) in enumerate(res.items()):
            rate = f"{r['recovery_rate']:.1f}% ({r['recoveries']}/{r['attempts']})"
            t = f"{r['avg_recovery_time_s']:.2f}" if r['avg_recovery_time_s'] else "N/A"
            print(f"{name if i == 0 else '':<22} {MODELS[name]['trained_on'] if i == 0 else '':<20} "
                  f"{iname:<15} {rate:<18} {t:<12}")
        print("-" * 80)
    out = Path("./analysis_results")
    out.mkdir(exist_ok=True)
    json.dump({"models_tested": list(results), "test_episodes_per_intensity": args.episodes, "results": results},
              open(out / "curriculum_comparison_results.json", "w"), indent=2)
    with open(out / "table_5_3_curriculum_comparison.tex", "w") as f:
        f.write("% Table 5.3: Cross-Level Performance (No Catastrophic Forgetting)\n\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Cross-Level Performance Evaluation}\n\\label{tab:curriculum_comparison}\n")
        f.write("\\begin{tabular}{llccc}\n\\toprule\n\\textbf{Model} & \\textbf{Test Intensity} & "
                "\\textbf{Recovery Rate} & \\textbf{Avg Time (s)} & \\textbf{Lowest (m)} \\\\\n\\midrule\n")
        for name, res in results.items():
            for i, (iname, r) in enumerate(res.items()):
                f.write(f"{name if i == 0 else ''} & {iname} & {r['recovery_rate']:.1f}\\% & "
                        f"{r['avg_recovery_time_s']:.2f} & {r['avg_min_altitude_m']:.1f} \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"\n💾 Results saved to: {out / 'curriculum_comparison_results.json'}")
    print(f"💾 LaTeX table: {out / 'table_5_3_curriculum_comparison.tex'}")
    print("\n✅ Analysis Complete!\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20, help="Test episodes per intensity level")
    ap.add_argument("--obs-norm", default="../Stage_1/runs/run1/models/obs_norm.npz")
    main(ap.parse_args())
