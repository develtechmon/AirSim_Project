r"""
TEST STAGE 2: HOVER IN WIND (13 OBSERVATIONS) - ArduPilot SITL + AirSim
=======================================================================

1. WHAT IT DOES
---------------
Flies ONE controller for 10 episodes x 500 steps (25 s each) in gusty wind and prints
TEST RESULTS, a verdict and the comparison to Stage 1. Fixed seeds: every controller and every
run sees the same start points and the same wind -> results can be compared side by side.

    | --model           | Who flies                                         | Role                |
    |-------------------|---------------------------------------------------|---------------------|
    | warm (default)    | Stage 2 PPO, started from Stage 1                 | MAIN RESULT         |
    | stage1            | Stage 1 network alone (no wind training)          | "before Stage 2"    |
    | ardupilot         | stock ArduPilot holding the point, no network     | industry baseline   |
    | pid               | Stage 1 teacher (P controller)                    | reference           |
    | cold              | Stage 2 PPO from random weights (if trained)      | no-curriculum check |


2. HOW TO RUN (AirSim + SITL in GPS mode running, in Stage_2\)
----------------------------------------------------------------
    py .\test_stage2_policy.py --model warm

    py .\test_stage2_policy.py --model stage1

    py .\test_stage2_policy.py --model ardupilot

    Defaults: 10 episodes, 500 steps, gusts 0-20 (AirSim wind setting).
    Optional: --episodes 5   --wind-strength 0 (calm-air hover check)   --stochastic (diagnosis)
    Time: about 5-6 min per model.


3. EXPECTED RESULT (gusts 0-20)
-------------------------------
    The top of the printout must say "Wind strength: 0-20.0 m/s" and episode lines show wind > 0.

    | --model    | Success | Avg distance | Meaning                                    |
    |------------|---------|--------------|--------------------------------------------|
    | warm       | ~100%   | ~0.2 m       | gust rejection learned                     |
    | stage1     | ~100%   | ~0.27 m      | drifts more: never trained in wind         |
    | ardupilot  | ~100%   | measured now | the bar PPO has to beat                    |

    | What you see                          | Action                                  |
    |---------------------------------------|-----------------------------------------|
    | warm closer than stage1 and ardupilot | Stage 2 claim holds                     |
    | warm about equal to ardupilot         | report honestly: PPO matches the stock  |
    | warm worse than stage1                | wrong model / no wind - send the JSON   |


4. OUTPUT FILES (folder made automatically)
-------------------------------------------
    | --model    | Saved to                                                        |
    |------------|-----------------------------------------------------------------|
    | warm       | runs\run<N>\results\warm_test\test_stage2_warm_<date>.csv/.json  |
    | stage1     | runs\run<N>\results\stage1_test\test_stage1_<date>.csv/.json     |
    | ardupilot  | runs\run<N>\results\ardupilot_test\test_ardupilot_<date>.csv/.json|
    | pid        | runs\run<N>\results\pid_test\test_pid_<date>.csv/.json           |
    | cold       | runs\run<N>\results\cold_test\test_stage2_cold_<date>.csv/.json  |
    New files get a new date stamp - nothing is overwritten.
    These are standalone checks. The THESIS table comes from collect_stage2_statistics.py.


5. DEFINITIONS (same as the paper)
----------------------------------
    Success          = episode reached the time limit (no out-of-bounds / altitude / tilt stop)
    Dist             = mean HORIZONTAL distance from the target over the episode
    Average Distance = mean of Dist over successful episodes
    Wind             = AirSim wind SETTING, not real-world m/s (20 = about 14 deg lean, see wind_check.py)
    Verdict:  >= 85% EXCELLENT, >= 70% GOOD, >= 50% MODERATE

    ArduPilot differences from the original AirSim version: the drone is not teleported between
    episodes (it flies to a random start, +/-1 m); Stage 1 comparison numbers are read from the
    Stage 1 result files, not typed in.
"""

import argparse
import csv
import datetime
import json

import numpy as np
from stable_baselines3 import PPO

from drone_hover_disturbance_env import DroneHoverDisturbanceEnv, run_folder, stage1_folder
from pid_expert import PIDExpertHover
from train_stage2_disturbance import latest_checkpoint

TEST_SEED0 = 1000          # gust episodes use seeds 1000, 1001, ...
SAVE = {                   # --model -> (folder inside runs\run<N>\results, file name start)
    "warm": ("warm_test", "test_stage2_warm"),
    "cold": ("cold_test", "test_stage2_cold"),
    "stage1": ("stage1_test", "test_stage1"),
    "pid": ("pid_test", "test_pid"),
    "ardupilot": ("ardupilot_test", "test_ardupilot"),
}


def load_controller(name, run_dir, s1, deterministic=True):
    """Returns (act(env, obs_norm) -> action, description)."""
    if name == "pid":
        teacher = PIDExpertHover()

        def act(env, obs):
            o = env._obs_raw()
            return teacher.get_action({"position": o[0:3], "velocity": o[3:6]})
        return act, "Stage 1 teacher (P controller)"
    if name == "ardupilot":
        def act(env, obs):
            return None                     # env sends the target position; ArduPilot holds it alone
        return act, "Stock ArduPilot (GUIDED position target, its own position controller, no network)"
    if name == "stage1":
        path = s1 / "models" / "stage1_ppo_init.zip"
    else:
        d = run_dir / "models" / name
        path = d / "hover_disturbance_policy.zip"
        if not path.exists():
            path = latest_checkpoint(d / "checkpoints")
            if path is None:
                raise SystemExit(f"No '{name}' model or checkpoint in {d} - train it first")
            print(f"\n⚠️  Final '{name}' model not found. Using latest checkpoint:\n   {path}\n")
    model = PPO.load(str(path), device="cpu")

    def act(env, obs):
        return model.predict(obs, deterministic=deterministic)[0]
    return act, str(path) + ("" if deterministic else " (STOCHASTIC: actions sampled as in training)")


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
    succ = [e for e in ok if e["success"]]
    return {"episodes": len(ok), "successes": len(succ), "success_pct": 100.0 * len(succ) / len(ok),
            "avg_distance_m": float(np.mean([e["ep_mean_hdist"] for e in succ])) if succ else None,
            "avg_wind_handled": float(np.mean([e["ep_mean_wind"] for e in succ])) if succ else None,
            "max_wind_survived": float(np.max([e["ep_max_wind"] for e in succ])) if succ else None,
            "avg_length_steps": float(np.mean(g("ep_steps"))),
            "rmse_m": float(np.mean(g("ep_rmse"))), "mean_dist_3d_m": float(np.mean(g("ep_mean_dist"))),
            "mean_dist_m": float(np.mean(g("ep_mean_hdist"))), "end_dist_m": float(np.mean(g("ep_end_dist"))),
            "max_dist_m": float(np.mean(g("ep_max_hdist"))), "worst_dist_m": float(np.max(g("ep_max_hdist"))),
            "hold_pct": float(np.mean(g("ep_hold_pct"))),
            "mean_wind_mps": float(np.mean(g("ep_mean_wind"))), "max_wind_mps": float(np.max(g("ep_max_wind"))),
            "fail_reasons": {r: g("reason").count(r) for r in set(g("reason")) if r != "timeout"}}


def stage1_reference(s1):
    """Stage 1 numbers from the Stage 1 result files (no wind)."""
    files = sorted((s1 / "results").glob("statistics_summary_*.json"))
    if not files:
        return None
    p = json.load(open(files[-1]))["policy"]
    return {"success": p.get("success_rate"), "dist": p.get("dist_mean"), "file": files[-1].name}


def main(args):
    run_dir = run_folder(args.run)
    s1 = stage1_folder(args.stage1_run)
    print("\n" + "=" * 70)
    print("🧪 TESTING STAGE 2: DISTURBANCE RECOVERY (13 OBSERVATIONS)")
    print("=" * 70)
    print({"pid": "Testing PID teacher with WIND disturbances!",
           "ardupilot": "Testing STOCK ARDUPILOT (no network) with WIND disturbances!"}.get(
        args.model, "Testing neural network with WIND disturbances!"))
    print()
    act, desc = load_controller(args.model, run_dir, s1, deterministic=not args.stochastic)
    print(f"[1/3] Loading model: {desc}")
    env = DroneHoverDisturbanceEnv(s1 / "models" / "obs_norm.npz", max_steps=args.max_steps,
                                   wind_strength=args.wind_strength)
    print("   ✅ Model loaded")
    print("   📊 Observation space: 13")
    print()
    print(f"[2/3] Running {args.episodes} test episodes...")
    print(f"   Wind strength: 0-{args.wind_strength} m/s")
    print(f"   Max steps: {args.max_steps} per episode")
    print()
    print("=" * 70)
    folder, prefix = SAVE[args.model]
    out = run_dir / "results" / folder
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out / f"{prefix}{'_stochastic' if args.stochastic else ''}_{stamp}"
    eps = []
    try:
        for i in range(args.episodes):
            e = run_episode(env, act, TEST_SEED0 + i)
            eps.append(e)
            print(f"Episode {i + 1:2d}/{args.episodes} | Steps: {e['ep_steps']:3d} | "
                  f"Success: {'✅' if e['success'] else '❌'} | Dist: {e['ep_mean_hdist']:.2f}m | "
                  f"Wind: {e['ep_mean_wind']:.1f}m/s (max: {e['ep_max_wind']:.1f}) | Reason: {e['reason']}")
    except KeyboardInterrupt:
        print("\nStopped by user - saving what we have")
    print("=" * 70)

    s = summarize(eps)
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)
    if s:
        print(f"Success Rate: {s['success_pct']:.0f}% ({s['successes']}/{s['episodes']} episodes)")
        if s["successes"]:
            print(f"Average Distance: {s['avg_distance_m']:.2f}m (successful episodes)")
            print(f"Average Wind Handled: {s['avg_wind_handled']:.1f} m/s")
            print(f"Maximum Wind Survived: {s['max_wind_survived']:.1f} m/s")
        print(f"Average Episode Length: {s['avg_length_steps']:.1f} steps")
        print()
        if s["fail_reasons"]:
            print("Failure Reasons:")
            for r, n in s["fail_reasons"].items():
                print(f"   - {r}: {n}")
            print()
        print("=" * 70)
        sr = s["success_pct"]
        if sr >= 85:
            print("✅ EXCELLENT! Policy handles wind disturbances very well!")
            if args.model in ("warm", "cold"):
                print("   Ready for Stage 3 (flip recovery)")
        elif sr >= 70:
            print("✅ GOOD! Policy handles wind reasonably well.")
            print("   Can proceed to Stage 3 or train longer")
        elif sr >= 50:
            print("⚠️  MODERATE! Policy partially learned disturbance recovery.")
            print("   Recommend: Train longer (750k timesteps)")
        else:
            print("❌ NEEDS IMPROVEMENT! Policy struggles with wind.")
            print("   Action needed:")
            print("   1. Train for more timesteps (1M)")
            print("   2. Reduce wind strength for initial training")
            print("   3. Check the warm-start PASS line at the start of training")
        print("=" * 70 + "\n")

        ref = stage1_reference(s1)
        print("=" * 70)
        print("📊 COMPARISON TO STAGE 1")
        print("=" * 70)
        if ref:
            print(f"Stage 1 (no wind):  {ref['success']:.0f}% success, ~{ref['dist']:.2f}m avg distance")
        else:
            print("Stage 1 (no wind):  no Stage 1 statistics file found")
        d = f"{s['avg_distance_m']:.2f}m" if s["successes"] else "-"
        label = {"warm": "Stage 2", "cold": "Stage 2 cold", "stage1": "Stage 1 network", "pid": "PID teacher",
                 "ardupilot": "Stock ArduPilot"}[args.model]
        print(f"{label + ' (with wind):':<20} {sr:.0f}% success, {d} avg distance")
        print()
        print("✅ Successfully maintained hover ability despite wind!" if sr >= 70
              else "⚠️  Wind significantly impacted performance")
        print("=" * 70 + "\n")

    cols = ["seed", "reason", "success", "ep_steps", "start_dist", "ep_mean_hdist", "ep_max_hdist",
            "ep_rmse", "ep_mean_dist", "ep_end_dist", "ep_hold_pct", "ep_mean_wind", "ep_max_wind"]
    with open(f"{base}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in eps:
            w.writerow([round(e[c], 3) if isinstance(e[c], float) else e[c] for c in cols])
    json.dump({"model": args.model, "controller": desc, "run": run_dir.name, "settings": vars(args),
               "summary": s, "stage1_reference": stage1_reference(s1),
               "episodes": eps}, open(f"{base}.json", "w"), indent=2, default=str)
    print(f"Saved: {base}.csv / .json")
    env.land()
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="warm", choices=list(SAVE))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--wind-strength", type=float, default=20.0)
    ap.add_argument("--run", default=None)
    ap.add_argument("--stage1-run", default=None)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions like during training (diagnosis only; the paper uses deterministic)")
    main(ap.parse_args())
