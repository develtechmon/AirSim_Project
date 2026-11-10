"""
TEST STAGE 2 POLICY IN WIND (ArduPilot SITL + AirSim, 13 observations)
======================================================================
Same job as your AirSim test_stage2_policy.py: one controller flies 10 episodes
in gusty wind 0-5 m/s (fixed seeds, so every test sees the same wind).

Controllers (--model):
    warm     Stage 2 PPO that started from Stage 1         (runs\\run<N>\\models\\warm)
    cold     Stage 2 PPO that started from random weights  (runs\\run<N>\\models\\cold)
    stage1   the Stage 1 network alone, no wind training   (..\\stage_1\\runs\\run<N>\\models)
    pid      the Stage 1 teacher (simple P controller)
If a PPO run is not finished, its newest checkpoint is used (and printed).

Success = survives all 500 steps (no out-of-bounds / height / tilt stop), as in the original.
Also reported: RMSE distance, max distance, % of time within 0.5 m ("holding").

Usage:
    py .\\test_stage2_policy.py --model warm --episodes 10
Saves runs\\run<N>\\results\\test_stage2_<model>_<date>.csv / .json
"""

import argparse
import csv
import datetime
import json

import numpy as np
from stable_baselines3 import PPO

from drone_hover_disturbance_env import STABLE_M, DroneHoverDisturbanceEnv, run_folder, stage1_folder
from pid_expert import PIDExpertHover
from train_stage2_disturbance import latest_checkpoint

TEST_SEED0 = 1000          # gust episodes use seeds 1000, 1001, ...


def load_controller(name, run_dir, s1):
    """Returns (act(env, obs_norm) -> action, description)."""
    if name == "pid":
        teacher = PIDExpertHover()

        def act(env, obs):
            o = env._obs_raw()
            return teacher.get_action({"position": o[0:3], "velocity": o[3:6]})
        return act, "Stage 1 teacher (P controller)"
    if name == "stage1":
        path = s1 / "models" / "stage1_ppo_init.zip"
    else:
        d = run_dir / "models" / name
        path = d / "hover_disturbance_policy.zip"
        if not path.exists():
            path = latest_checkpoint(d / "checkpoints")
            if path is None:
                raise SystemExit(f"No '{name}' model or checkpoint in {d} - train it first")
            print(f"   NOTE: '{name}' training not finished - using checkpoint {path.name}")
    model = PPO.load(str(path), device="cpu")

    def act(env, obs):
        return model.predict(obs, deterministic=True)[0]
    return act, str(path)


def run_episode(env, act, seed, fixed_wind=None):
    obs, info0 = env.reset(options={"seed": seed, "fixed_wind": fixed_wind})
    while True:
        obs, r, term, trunc, info = env.step(act(env, obs))
        if term or trunc:
            break
    info["start_dist"] = info0["start_dist"]
    info["seed"] = seed
    info["fixed_wind"] = fixed_wind
    info["success"] = info["reason"] == "timeout"
    return info


def summarize(eps):
    ok = [e for e in eps if e["reason"] != "update_pause"]
    if not ok:
        return {}
    g = lambda k: [e[k] for e in ok]
    return {"episodes": len(ok), "success_pct": 100.0 * np.mean(g("success")),
            "rmse_m": float(np.mean(g("ep_rmse"))), "mean_dist_m": float(np.mean(g("ep_mean_dist"))),
            "max_dist_m": float(np.mean(g("ep_max_dist"))), "worst_dist_m": float(np.max(g("ep_max_dist"))),
            "end_dist_m": float(np.mean(g("ep_end_dist"))), "hold_pct": float(np.mean(g("ep_hold_pct"))),
            "mean_wind_mps": float(np.mean(g("ep_mean_wind"))), "max_wind_mps": float(np.max(g("ep_max_wind"))),
            "fail_reasons": {r: g("reason").count(r) for r in set(g("reason")) if r != "timeout"}}


def main(args):
    run_dir = run_folder(args.run)
    s1 = stage1_folder(args.stage1_run)
    print("\n" + "=" * 70)
    print(f"TESTING STAGE 2 IN WIND - {args.model.upper()} - {run_dir.name.upper()}")
    print("=" * 70)
    act, desc = load_controller(args.model, run_dir, s1)
    print(f"Controller: {desc}")
    print(f"Wind: gusts 0-{args.wind_strength} m/s, seeds {TEST_SEED0}..{TEST_SEED0 + args.episodes - 1}")
    print("=" * 70)
    env = DroneHoverDisturbanceEnv(s1 / "models" / "obs_norm.npz", max_steps=args.max_steps,
                                   wind_strength=args.wind_strength)
    out = run_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out / f"test_stage2_{args.model}_{stamp}"
    eps = []
    try:
        for i in range(args.episodes):
            e = run_episode(env, act, TEST_SEED0 + i)
            eps.append(e)
            print(f"Episode {i + 1:2d}/{args.episodes} | steps {e['ep_steps']:3d} | "
                  f"{'OK  ' if e['success'] else 'FAIL'} | RMSE {e['ep_rmse']:.2f} m | "
                  f"max {e['ep_max_dist']:.2f} m | held {e['ep_hold_pct']:3.0f}% | "
                  f"wind {e['ep_mean_wind']:.1f} m/s (max {e['ep_max_wind']:.1f}) | {e['reason']}")
    except KeyboardInterrupt:
        print("\nStopped by user - saving what we have")
    finally:
        s = summarize(eps)
        cols = ["seed", "reason", "success", "ep_steps", "start_dist", "ep_rmse", "ep_mean_dist",
                "ep_max_dist", "ep_end_dist", "ep_hold_pct", "ep_mean_wind", "ep_max_wind"]
        with open(f"{base}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for e in eps:
                w.writerow([e[c] if not isinstance(e[c], float) else round(e[c], 3) for c in cols])
        json.dump({"model": args.model, "controller": desc, "run": run_dir.name, "settings": vars(args),
                   "summary": s, "episodes": eps}, open(f"{base}.json", "w"), indent=2, default=str)
        print("=" * 70)
        if s:
            print(f"Success rate:  {s['success_pct']:.0f}% ({s['episodes']} episodes, survive all steps)")
            print(f"RMSE distance: {s['rmse_m']:.2f} m | mean {s['mean_dist_m']:.2f} m | "
                  f"max (avg) {s['max_dist_m']:.2f} m | worst {s['worst_dist_m']:.2f} m")
            print(f"Held within {STABLE_M} m: {s['hold_pct']:.0f}% of the time")
            print(f"Wind: mean {s['mean_wind_mps']:.1f} m/s, max {s['max_wind_mps']:.1f} m/s")
            if s["fail_reasons"]:
                print(f"Failures: {s['fail_reasons']}")
        print(f"Saved: {base}.csv / .json")
        print("=" * 70)
        env.land()
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["warm", "cold", "stage1", "pid"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--wind-strength", type=float, default=5.0)
    ap.add_argument("--run", default=None)
    ap.add_argument("--stage1-run", default=None)
    main(ap.parse_args())
