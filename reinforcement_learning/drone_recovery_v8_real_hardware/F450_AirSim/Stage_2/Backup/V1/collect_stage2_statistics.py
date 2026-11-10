"""
STAGE 2 STATISTICS: WHO HOLDS POSITION BEST IN WIND? (ArduPilot SITL + AirSim)
==============================================================================
All controllers fly IDENTICAL conditions (same start points, same wind, fixed seeds):

    pid      Stage 1 teacher (P controller)              - reference
    stage1   Stage 1 network alone (no wind training)    - "before Stage 2"
    warm     Stage 2 PPO started from Stage 1            - curriculum
    cold     Stage 2 PPO started from random weights     - no curriculum

Conditions:
    Part A  gusts 0-20 m/s (like training), 10 episodes, seeds 2000..2009
    Part B  constant wind 5, 10, 15, 20 m/s (random direction), 2 episodes each, seeds 3000..

A controller whose model does not exist yet is skipped (e.g. run before cold is trained).

Usage (AirSim + SITL in GPS mode):
    py .\\collect_stage2_statistics.py
    py .\\collect_stage2_statistics.py --models stage1 warm      (only some)
Saves runs\\run<N>\\results\\stage2_statistics_<date>.json / .pkl / _report.txt
About 45 min for all four (each episode 25 s + ~5 s reset).
"""

import argparse
import datetime
import json
import pickle


from drone_hover_disturbance_env import STABLE_M, DroneHoverDisturbanceEnv, run_folder, stage1_folder
from test_stage2_policy import load_controller, run_episode, summarize

GUST_SEED0 = 2000
LEVEL_SEED0 = 3000
LEVELS = [5.0, 10.0, 15.0, 20.0]
NAMES = {"pid": "PID teacher", "stage1": "Stage 1 (BC)", "warm": "PPO warm", "cold": "PPO cold"}


def report(res, args, run_dir):
    ctrls = [c for c in args.models if c in res]
    L = ["=" * 86, "STAGE 2 STATISTICS - POSITION HOLD IN WIND (ArduPilot SITL + AirSim)",
         f"Run: {run_dir.name} | episodes: {args.max_steps} steps at 20 Hz | identical seeds for all controllers",
         "=" * 86, "", f"TABLE 1 - GUSTS 0-{args.wind_strength:.0f} m/s (m/s, {args.gust_episodes} episodes)", "-" * 86,
         f"{'Controller':<16}{'Success %':>10}{'RMSE m':>9}{'Dist m':>9}{'Max m':>9}{'Worst m':>9}"
         f"{'End m':>8}{'Held %':>8}{'Wind':>9}", "-" * 86]
    for c in ctrls:
        s = res[c]["gust_summary"]
        if s:
            L.append(f"{NAMES[c]:<16}{s['success_pct']:>10.0f}{s['rmse_m']:>9.3f}{s['mean_dist_m']:>9.3f}"
                     f"{s['max_dist_m']:>9.3f}{s['worst_dist_m']:>9.3f}{s['end_dist_m']:>8.3f}"
                     f"{s['hold_pct']:>8.0f}{s['mean_wind_mps']:>9.2f}")
    L += ["-" * 86, f"Dist/Max/Worst = horizontal distance (as in the paper); RMSE = 3D. "
          f"Held % = time within {STABLE_M} m. Success = reached timeout.",
          "", "TABLE 2 - CONSTANT WIND: RMSE (m) / SUCCESS (%)", "-" * 86,
          f"{'Controller':<16}" + "".join(f"{f'{l:.0f} m/s':>14}" for l in LEVELS), "-" * 86]
    for c in ctrls:
        cells = []
        for l in LEVELS:
            s = res[c]["level_summary"].get(str(l), {})
            cells.append(f"{s['rmse_m']:.3f}/{s['success_pct']:.0f}" if s else "-")
        L.append(f"{NAMES[c]:<16}" + "".join(f"{x:>14}" for x in cells))
    L += ["-" * 86, "", "TABLE 3 - EFFECT OF STAGE 2 (gusts, RMSE)", "-" * 86]
    g = {c: res[c]["gust_summary"].get("rmse_m") for c in ctrls if res[c]["gust_summary"]}
    for a, b, text in (("warm", "stage1", "Stage 2 warm vs Stage 1 alone"),
                       ("warm", "cold", "Warm start vs cold start"),
                       ("warm", "pid", "Stage 2 warm vs PID teacher")):
        if a in g and b in g:
            L.append(f"  {text:<36} RMSE {g[a]:.3f} vs {g[b]:.3f} m -> "
                     f"{100 * (g[b] - g[a]) / g[b]:+.1f}% (positive = {NAMES[a]} better)")
    L += ["=" * 86]
    return "\n".join(L)


def main(args):
    run_dir = run_folder(args.run)
    s1 = stage1_folder(args.stage1_run)
    print("\n" + "=" * 70)
    print(f"STAGE 2 STATISTICS - {run_dir.name.upper()}")
    print("=" * 70)
    ctrls = {}
    for c in args.models:
        try:
            ctrls[c] = load_controller(c, run_dir, s1)
            print(f"  {NAMES[c]:<14} {ctrls[c][1]}")
        except SystemExit as e:
            print(f"  {NAMES[c]:<14} SKIPPED - {e}")
    env = DroneHoverDisturbanceEnv(s1 / "models" / "obs_norm.npz", max_steps=args.max_steps,
                                   wind_strength=args.wind_strength)
    res = {}
    try:
        for c, (act, desc) in ctrls.items():
            print(f"\n--- {NAMES[c]} ---")
            gust = []
            for i in range(args.gust_episodes):
                e = run_episode(env, act, GUST_SEED0 + i)
                gust.append(e)
                print(f"  gust  ep {i + 1:2d} | {'OK  ' if e['success'] else 'FAIL'} | RMSE {e['ep_rmse']:.2f} m | "
                      f"max {e['ep_max_dist']:.2f} m | wind {e['ep_mean_wind']:.1f} m/s | {e['reason']}")
            levels = {}
            for j, l in enumerate(LEVELS):
                levels[str(l)] = []
                for k in range(args.level_episodes):
                    e = run_episode(env, act, LEVEL_SEED0 + 10 * j + k, fixed_wind=l)
                    levels[str(l)].append(e)
                    print(f"  {l:.0f} m/s ep {k + 1} | {'OK  ' if e['success'] else 'FAIL'} | "
                          f"RMSE {e['ep_rmse']:.2f} m | max {e['ep_max_dist']:.2f} m | {e['reason']}")
            res[c] = {"controller": desc, "gust": gust, "levels": levels,
                      "gust_summary": summarize(gust),
                      "level_summary": {k: summarize(v) for k, v in levels.items()}}
    except KeyboardInterrupt:
        print("\nStopped by user - saving what we have")
    finally:
        env.land()
        env.close()
    out = run_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out / f"stage2_statistics_{stamp}"
    meta = {"date": stamp, "run": run_dir.name, "stage1_run": str(s1), "settings": vars(args),
            "levels_mps": LEVELS, "names": NAMES}
    with open(f"{base}.pkl", "wb") as f:
        pickle.dump({**meta, "results": res}, f)
    light = {c: {k: v for k, v in r.items() if k in ("controller", "gust_summary", "level_summary")}
             for c, r in res.items()}
    json.dump({**meta, "results": light}, open(f"{base}.json", "w"), indent=2, default=str)
    text = report(res, args, run_dir)
    with open(f"{base}_report.txt", "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {base}.json / .pkl / _report.txt")
    print("Next: py .\\make_stage2_summary.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["pid", "stage1", "warm", "cold"],
                    choices=["pid", "stage1", "warm", "cold"])
    ap.add_argument("--gust-episodes", type=int, default=10)
    ap.add_argument("--level-episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--wind-strength", type=float, default=20.0)
    ap.add_argument("--run", default=None)
    ap.add_argument("--stage1-run", default=None)
    main(ap.parse_args())
