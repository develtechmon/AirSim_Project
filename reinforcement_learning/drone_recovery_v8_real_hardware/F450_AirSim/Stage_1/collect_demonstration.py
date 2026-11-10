"""
COLLECT EXPERT DEMONSTRATIONS (ArduPilot SITL + AirSim)
=======================================================
Same job as your AirSim collect_demonstration.py:
PID expert flies the drone, we save state-action pairs for behavioral cloning.

Differences from the AirSim version:
  - talks to ArduPilot through ap_link.py (same code later on the real drone)
  - fixed 20 Hz loop, real timing measured
  - observation uses position RELATIVE to target (works at any altitude)
  - each episode = new random target near the drone (no reset needed)

Usage:
    python collect_demonstration.py --episodes 5 --run practice      (quick test)
    python collect_demonstration.py --episodes 200 --steps 300       (full run -> NEW run folder)

Every full run gets its own numbered folder, made automatically:
    runs/run1/   runs/run2/   runs/run3/ ...   (highest number = latest)
All later scripts (train, test, statistics, rmse, vd, summary) use the
latest run folder automatically, or the one you pick with --run N.

Saves in runs/run<N>/demonstrations/:
    expert_demonstrations_<sensor>_<date>.pkl     the data
    expert_demonstrations_<sensor>_<date>.csv     one line per episode
    expert_demonstrations_<sensor>_<date>.json    settings, firmware, results
"""

import argparse
import csv
import datetime
import json
import pickle
import time
from pathlib import Path

import numpy as np

from ap_link import APLink, RateKeeper
from pid_expert import EXPERT_DESCRIPTION, PIDExpertHover
from train_imitation import run_folder

ABORT_ALT = 0.3      # m (indoor: rangefinder height)
ABORT_TILT = 45.0    # deg
FENCE_M = 1.5        # m beyond the target box -> safety stop (keeps it off walls)
SETTLED_M = 0.3      # end error below this = settled

# saved with every run, so you always know how the data was made
SNAPSHOT_PARAMS = [
    "EK3_SRC1_POSXY", "EK3_SRC1_VELXY", "EK3_SRC1_POSZ", "EK3_SRC1_VELZ", "EK3_SRC1_YAW",
    "FLOW_TYPE", "SIM_FLOW_ENABLE", "RNGFND1_TYPE", "RNGFND1_MAX_CM",
    "GPS_TYPE", "GUID_OPTIONS", "ARMING_CHECK", "WPNAV_SPEED", "WPNAV_SPEED_UP", "WPNAV_SPEED_DN",
]


def sensor_mode(params):
    if params.get("EK3_SRC1_VELXY") == 5:
        return "optflow"
    if params.get("EK3_SRC1_POSXY") == 3:
        return "gps"
    return "unknown"


def collect_demonstrations(args):
    np.random.seed(0)
    link = APLink(args.connect)
    link.wait_state()
    firmware = link.firmware_version()
    params = link.get_params(SNAPSHOT_PARAMS)
    mode = sensor_mode(params)

    run_dir = run_folder(args.run, new=True)
    save_dir = run_dir / "demonstrations"
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = save_dir / f"expert_demonstrations_{mode}_{stamp}"

    print("\n" + "=" * 70)
    print(f"COLLECTING EXPERT DEMONSTRATIONS (ArduPilot) - {run_dir.name.upper()}")
    print("=" * 70)
    print(f"Run folder: {run_dir}")
    print(f"Episodes: {args.episodes} x {args.steps} steps at {args.hz:.0f} Hz")
    print(f"Firmware: {firmware} | sensor mode: {mode}")
    print(f"Saving to: {base}.pkl / .csv / .json")
    print("=" * 70 + "\n")

    link.wait_state()
    if not link.armed:
        link.takeoff_nogps(0.5 * (args.alt_min + args.alt_max))
    else:
        link.set_mode("GUIDED")

    # HOME = where the drone hovers after takeoff. Every target is picked around
    # HOME (not around where the drone happens to be), so it never wanders away.
    link.update()
    home = link.pos.copy()
    home_alt = 0.5 * (args.alt_min + args.alt_max)
    fence = args.xy_offset + FENCE_M
    print(f"Home: N {home[0]:.2f} E {home[1]:.2f} m | targets within +/-{args.xy_offset} m of home, "
          f"{home_alt:.1f} +/- {args.z_offset} m high | safety fence {fence:.1f} m")

    expert = PIDExpertHover(dt=1.0 / args.hz)
    states, actions, ep_ids, targets, ep_rows = [], [], [], [], []
    rk = RateKeeper(args.hz)
    t0 = time.time()
    aborted = False

    csv_f = open(f"{base}.csv", "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["episode", "target_alt_m", "start_err_m", "end_err_m", "rmse_m",
                    "settled", "max_cmd_mps", "saturated_steps", "steps"])

    try:
        for ep in range(args.episodes):
            # new target around HOME
            link.update()
            alt = np.clip(home_alt + np.random.uniform(-args.z_offset, args.z_offset),
                          args.alt_min, args.alt_max)
            target = np.array([home[0] + np.random.uniform(-args.xy_offset, args.xy_offset),
                               home[1] + np.random.uniform(-args.xy_offset, args.xy_offset),
                               -alt])
            targets.append(target)
            expert.reset()
            errs, acts = [], []

            for _ in range(args.steps):
                link.update()
                away = float(np.linalg.norm(link.pos[0:2] - home[0:2]))
                if link.alt < ABORT_ALT or link.tilt_deg > ABORT_TILT or away > fence:
                    print(f"ABORT: alt={link.alt:.2f} m tilt={link.tilt_deg:.1f} deg "
                          f"distance from home={away:.2f} m")
                    aborted = True
                    break
                obs = link.obs(target)
                action = expert.get_action({"position": obs[0:3], "velocity": obs[3:6]})
                link.send_velocity(*action)
                states.append(obs)
                actions.append(action)
                ep_ids.append(ep)
                errs.append(float(np.linalg.norm(obs[0:3])))
                acts.append(action)
                rk.wait()

            if errs:
                a = np.array(acts)
                end_err = float(np.mean(errs[-20:]))
                row = [ep, round(alt, 2), round(errs[0], 3), round(end_err, 3),
                       round(float(np.sqrt(np.mean(np.square(errs)))), 3),
                       int(end_err < SETTLED_M), round(float(np.abs(a).max()), 2),
                       int((np.abs(a) >= 4.99).any(axis=1).sum()), len(errs)]
                csv_w.writerow(row)
                csv_f.flush()          # written live - safe if it crashes
                ep_rows.append(row)

            if aborted:
                break
            if (ep + 1) % 10 == 0 or ep < 3:
                s = rk.stats()
                print(f"Episode {ep+1:4d}/{args.episodes} | start {ep_rows[-1][2]:.2f} m -> "
                      f"end {ep_rows[-1][3]:.2f} m | loop {s['dt_mean']*1000:.1f} ms | "
                      f"overruns {s['overruns']} | {(time.time()-t0)/60:.1f} min")

    except KeyboardInterrupt:
        print("\nStopped by user - saving what we have")

    finally:
        link.hold()
        csv_f.close()
        timing = rk.stats()
        settled = float(np.mean([r[5] for r in ep_rows]) * 100) if ep_rows else 0.0
        end_err_mean = float(np.mean([r[3] for r in ep_rows])) if ep_rows else float("nan")

        with open(f"{base}.pkl", "wb") as f:
            pickle.dump({
                "states": np.array(states, dtype=np.float32),
                "actions": np.array(actions, dtype=np.float32),
                "episode_ids": np.array(ep_ids),
                "targets": np.array(targets),
            }, f)

        info = {
            "firmware": firmware, "sensor_mode": mode, "params": params,
            "settings": vars(args),
            "obs": "rel_pos(3, NED, pos-target) | vel(3, NED) | quat wxyz(4) | body rates(3)",
            "action": "velocity command vx, vy, vz (NED, m/s), GUIDED",
            "expert": EXPERT_DESCRIPTION,
            "samples": len(states), "episodes_done": len(ep_rows), "aborted": aborted,
            "timing": timing,
            "settled_pct": settled, "settled_threshold_m": SETTLED_M,
            "end_err_mean_m": end_err_mean,
        }
        with open(f"{base}.json", "w") as f:
            json.dump(info, f, indent=2, default=str)

        print("\n" + "=" * 70)
        print("SAVED")
        print("=" * 70)
        print(f"Data:      {base}.pkl")
        print(f"Samples:   {len(states):,} | episodes {len(ep_rows)}")
        print(f"Loop time: {timing['dt_mean']*1000:.1f} ms (target {1000/args.hz:.1f}) | "
              f"max {timing['dt_max']*1000:.1f} ms | overruns {timing['overruns']}")
        print(f"Settled:   {settled:.0f}% of episodes (end error < {SETTLED_M} m) | "
              f"mean end error {end_err_mean:.2f} m")
        link.land()                      # data is already saved - now land automatically
        print("=" * 70)
        print(f"Run:       {run_dir}")
        print(f"\nNext step: python train_imitation.py"
              + (" --run practice" if args.run == "practice" else f"   (uses {run_dir.name})"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--connect", default="tcp:127.0.0.1:5762")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--alt-min", type=float, default=1.0)
    parser.add_argument("--alt-max", type=float, default=3.0)
    parser.add_argument("--xy-offset", type=float, default=1.0)
    parser.add_argument("--z-offset", type=float, default=0.5)
    parser.add_argument("--run", default=None,
                        help="empty = NEW run folder (run1, run2, ...) | 'practice' = quick test folder")
    collect_demonstrations(parser.parse_args())
