"""
TEST HOVER POLICY (ArduPilot pipeline, 13 observations)
=======================================================
Same job as your AirSim test_hover_policy.py:
the NEURAL NETWORK flies alone (not the PID) - did it learn to hover?

Each episode: new random target near the drone (same scheme as collection,
fixed seed so every test run uses the same targets).

Success = no safety stop AND end error < 0.5 m
Settled = end error < 0.3 m (same as collection)

Usage:
    python test_hover_policy.py --episodes 10              (latest run folder)
    python test_hover_policy.py --episodes 10 --run 1      (pick run 1)

Loads runs/run<N>/models/ and saves in runs/run<N>/results/:
    test_hover_<date>.csv    one line per episode
    test_hover_<date>.json   summary
    test_hover_<date>.pkl    every step (errors, actions) for plots
"""

import argparse
import csv
import datetime
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from ap_link import APLink, RateKeeper
from train_imitation import ACT_LIMIT, HoverPolicy, normalize, run_folder

ABORT_ALT = 0.3      # m (indoor: rangefinder height)
ABORT_TILT = 45.0
FENCE_M = 1.5        # m beyond the target box -> safety stop (keeps it off walls)
SUCCESS_M = 0.5
SETTLED_M = 0.3


def test_policy(args):
    run_dir = run_folder(args.run)
    model_dir = run_dir / "models"
    model = HoverPolicy()
    model.load_state_dict(torch.load(model_dir / "hover_policy_best.pth", map_location="cpu"))
    model.eval()
    norm = np.load(model_dir / "obs_norm.npz")
    mean, std = norm["mean"], norm["std"]

    print("\n" + "=" * 70)
    print(f"TESTING LEARNED HOVER POLICY (ArduPilot) - {run_dir.name.upper()}")
    print("=" * 70)
    print(f"Run folder: {run_dir}")
    print("The NEURAL NETWORK flies - not the PID expert")
    print(f"Episodes: {args.episodes} x {args.steps} steps at {args.hz:.0f} Hz\n")

    link = APLink(args.connect)
    link.wait_state()
    firmware = link.firmware_version()
    link.wait_state()
    if not link.armed:
        link.takeoff_nogps(0.5 * (args.alt_min + args.alt_max))
    else:
        link.set_mode("GUIDED")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = run_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"test_hover_{stamp}"
    csv_f = open(f"{base}.csv", "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["episode", "start_err_m", "end_err_m", "rmse_m", "max_err_m",
                    "success", "settled", "reason"])

    link.update()
    home = link.pos.copy()                   # targets picked around HOME, never drifts away
    home_alt = 0.5 * (args.alt_min + args.alt_max)
    fence = args.xy_offset + FENCE_M
    print(f"Home: N {home[0]:.2f} E {home[1]:.2f} m | safety fence {fence:.1f} m")
    rng = np.random.RandomState(42)          # same targets every test run
    rk = RateKeeper(args.hz)
    results, traj = [], []
    print("=" * 70)
    try:
        for ep in range(args.episodes):
            link.update()
            alt = np.clip(home_alt + rng.uniform(-args.z_offset, args.z_offset),
                          args.alt_min, args.alt_max)
            target = np.array([home[0] + rng.uniform(-args.xy_offset, args.xy_offset),
                               home[1] + rng.uniform(-args.xy_offset, args.xy_offset), -alt])
            errs, acts, reason = [], [], "completed"

            for _ in range(args.steps):
                link.update()
                if (link.alt < ABORT_ALT or link.tilt_deg > ABORT_TILT
                        or np.linalg.norm(link.pos[0:2] - home[0:2]) > fence):
                    reason = "safety_stop"
                    break
                obs = link.obs(target)
                with torch.no_grad():
                    a = model(torch.as_tensor(normalize(obs, mean, std)).unsqueeze(0))
                action = np.clip(a.squeeze(0).numpy(), -ACT_LIMIT, ACT_LIMIT)
                link.send_velocity(*action)
                errs.append(float(np.linalg.norm(obs[0:3])))
                acts.append(action.tolist())
                rk.wait()

            end_err = float(np.mean(errs[-20:])) if errs else float("nan")
            success = reason == "completed" and end_err < SUCCESS_M
            row = [ep + 1, round(errs[0], 3), round(end_err, 3),
                   round(float(np.sqrt(np.mean(np.square(errs)))), 3),
                   round(max(errs), 3), int(success), int(end_err < SETTLED_M), reason]
            csv_w.writerow(row)
            csv_f.flush()
            results.append(row)
            traj.append({"target": target, "errors": errs, "actions": acts})
            print(f"Episode {ep+1:2d}/{args.episodes} | start {row[1]:.2f} m -> end {row[2]:.2f} m"
                  f" | {'OK  ' if success else 'FAIL'} | {reason}")
            if reason == "safety_stop":
                link.hold()
                print("Safety stop - test ended. Check the drone before continuing.")
                break

    except KeyboardInterrupt:
        print("\nStopped by user - saving what we have")

    finally:
        link.hold()
        csv_f.close()
        n = len(results)
        success_rate = 100.0 * sum(r[5] for r in results) / n if n else 0.0
        settled_rate = 100.0 * sum(r[6] for r in results) / n if n else 0.0
        end_mean = float(np.mean([r[2] for r in results])) if n else float("nan")
        timing = rk.stats()

        with open(f"{base}.pkl", "wb") as f:
            pickle.dump(traj, f)
        with open(f"{base}.json", "w") as f:
            json.dump({
                "firmware": firmware, "model_dir": str(model_dir), "run": run_dir.name, "settings": vars(args),
                "episodes": n, "success_rate_pct": success_rate,
                "success_definition": f"no safety stop and end error < {SUCCESS_M} m",
                "settled_rate_pct": settled_rate, "settled_threshold_m": SETTLED_M,
                "end_err_mean_m": end_mean, "timing": timing,
            }, f, indent=2, default=str)

        print("=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(f"Success rate: {success_rate:.0f}% ({sum(r[5] for r in results)}/{n})"
              f"  [no safety stop and end error < {SUCCESS_M} m]")
        print(f"Settled:      {settled_rate:.0f}%  [end error < {SETTLED_M} m]")
        print(f"Mean end error: {end_mean:.2f} m")
        print(f"Loop time: {timing['dt_mean']*1000:.1f} ms | overruns {timing['overruns']}")
        print(f"Saved: {base}.csv / .json / .pkl")
        link.land()                      # data is already saved - now land automatically
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--connect", default="tcp:127.0.0.1:5762")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--alt-min", type=float, default=1.0)
    parser.add_argument("--alt-max", type=float, default=3.0)
    parser.add_argument("--xy-offset", type=float, default=1.0)
    parser.add_argument("--z-offset", type=float, default=0.5)
    parser.add_argument("--run", default=None, help="run number or 'practice' (default: latest run)")
    test_policy(parser.parse_args())
