r"""
STAGE 1 STATISTICS: PID EXPERT vs BC POLICY (ArduPilot SITL + AirSim)
=====================================================================

1. WHAT IT DOES
---------------
The PID expert (P teacher) and the trained BC network fly the SAME 10 targets back to back
(seed 42), so they can be compared fairly: RMSE, drift, end error, transfer efficiency.
Uses the existing trained model - no retraining.

    Transfer efficiency = (1 - (BC_RMSE - PID_RMSE) / PID_RMSE) x 100%


2. HOW TO RUN (AirSim + SITL in GPS mode running, in Stage_1\)
----------------------------------------------------------------
    py .\collect_stage1_statistics.py

    Defaults: 10 episodes per controller x 300 steps, latest run folder.  About 10 min.
    Then (no simulator needed):

    py .\rmse_baseline.py

    py .\vd_distribution.py

    py .\make_stage1_summary.py


3. EXPECTED RESULT (your run1)
------------------------------
    | Measure                          | Pass rule | 17 Oct  | 23 Oct (repeat) |
    |----------------------------------|-----------|---------|-----------------|
    | Position RMSE PID / BC (m)       |           | 0.344 / 0.348 | 0.344 / 0.348 |
    | Transfer efficiency, position    | >= 90%    | 98.9%   | 98.9%           |
    | Transfer efficiency, altitude    | >= 90%    | 97.3%   | 97.1%           |
    | Success PID / BC                 | 100%      | 100/100 | 100/100         |


4. OUTPUT FILES (folder made automatically)
-------------------------------------------
    runs\run<N>\results\statistics\statistics_summary_<date>.pkl   every step of every episode
    runs\run<N>\results\statistics\statistics_summary_<date>.json  all numbers (no per-step arrays)
    runs\run<N>\results\statistics\statistics_report_<date>.txt    readable tables
"""

import argparse
import datetime
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from ap_link import APLink, RateKeeper
from pid_expert import PIDExpertHover
from train_imitation import ACT_LIMIT, HoverPolicy, normalize, run_folder

ABORT_ALT = 0.3
ABORT_TILT = 45.0
FENCE_M = 1.5        # m beyond the target box -> safety stop (keeps it off walls)
SUCCESS_M = 0.5
SETTLED_M = 0.3


def histogram(values, bins=40):
    counts, edges = np.histogram(values, bins=bins)
    return {"x": (0.5 * (edges[:-1] + edges[1:])).tolist(), "y": counts.tolist(),
            "edges": edges.tolist(), "mean": float(np.mean(values)),
            "std": float(np.std(values)), "median": float(np.median(values))}


def run_controller(link, label, act_fn, reset_fn, args, home):
    """Fly one controller over the fixed target sequence (seed 42), targets around HOME."""
    print("\n" + "=" * 70)
    print(f"{label}")
    print("=" * 70)
    rng = np.random.RandomState(42)
    rk = RateKeeper(args.hz)
    episodes = []
    for ep in range(args.episodes):
        link.update()
        alt_t = float(np.clip(0.5 * (args.alt_min + args.alt_max)
                              + rng.uniform(-args.z_offset, args.z_offset),
                              args.alt_min, args.alt_max))
        target = np.array([home[0] + rng.uniform(-args.xy_offset, args.xy_offset),
                           home[1] + rng.uniform(-args.xy_offset, args.xy_offset), -alt_t])
        reset_fn()
        rec = {"target": target.tolist(), "target_alt": alt_t, "pos_err": [], "alt": [],
               "horiz": [], "actions": [], "reason": "completed"}
        for _ in range(args.steps):
            link.update()
            if (link.alt < ABORT_ALT or link.tilt_deg > ABORT_TILT
                    or np.linalg.norm(link.pos[0:2] - home[0:2]) > args.xy_offset + FENCE_M):
                rec["reason"] = "safety_stop"
                break
            obs = link.obs(target)
            a = act_fn(obs)
            link.send_velocity(*a)
            rec["pos_err"].append(float(np.linalg.norm(obs[0:3])))
            rec["alt"].append(float(link.alt))
            rec["horiz"].append(float(np.linalg.norm(obs[0:2])))
            rec["actions"].append([float(x) for x in a])
            rk.wait()

        err = np.array(rec["pos_err"]) if rec["pos_err"] else np.array([np.nan])
        alt_err = np.abs(np.array(rec["alt"]) - alt_t) if rec["alt"] else np.array([np.nan])
        rec["end_err"] = float(np.mean(err[-20:]))
        rec["pos_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        rec["alt_rmse"] = float(np.sqrt(np.mean(alt_err ** 2)))
        rec["alt_std"] = float(np.std(rec["alt"])) if rec["alt"] else float("nan")
        rec["dist_max"] = float(max(rec["horiz"])) if rec["horiz"] else float("nan")
        rec["dist_mean"] = float(np.mean(rec["horiz"])) if rec["horiz"] else float("nan")
        rec["success"] = rec["reason"] == "completed" and rec["end_err"] < SUCCESS_M
        rec["settled"] = rec["reason"] == "completed" and rec["end_err"] < SETTLED_M
        episodes.append(rec)
        print(f"  Episode {ep+1:2d}/{args.episodes} | target alt {alt_t:.2f} m | "
              f"pos RMSE {rec['pos_rmse']:.3f} m | alt RMSE {rec['alt_rmse']:.3f} m | "
              f"end {rec['end_err']:.2f} m | {'OK' if rec['success'] else 'FAIL'} | {rec['reason']}")
        if rec["reason"] == "safety_stop":
            print("  Safety stop - this controller's run ends here.")
            break
    return episodes, rk.stats()


def build_stats(label, episodes, timing):
    n = len(episodes)
    acts = np.array([a for e in episodes for a in e["actions"]])
    g = lambda k: [e[k] for e in episodes]
    return {
        "label": label, "n_episodes": n, "timing": timing,
        "pos_rmse_mean": float(np.mean(g("pos_rmse"))), "pos_rmse_std": float(np.std(g("pos_rmse"))),
        "alt_rmse_mean": float(np.mean(g("alt_rmse"))), "alt_rmse_std": float(np.std(g("alt_rmse"))),
        "alt_std_mean": float(np.mean(g("alt_std"))),
        "dist_mean": float(np.mean(g("dist_mean"))), "dist_max_mean": float(np.mean(g("dist_max"))),
        "dist_max_abs": float(np.max(g("dist_max"))),
        "end_err_mean": float(np.mean(g("end_err"))), "end_err_std": float(np.std(g("end_err"))),
        "success_rate": 100.0 * sum(g("success")) / n if n else 0.0,
        "settled_rate": 100.0 * sum(g("settled")) / n if n else 0.0,
        "ep_pos_rmse": g("pos_rmse"), "ep_alt_rmse": g("alt_rmse"), "ep_alt_std": g("alt_std"),
        "ep_dist_max": g("dist_max"), "ep_end_err": g("end_err"), "ep_reason": g("reason"),
        "target_alts": g("target_alt"),
        "action_mean": acts.mean(axis=0).tolist() if len(acts) else [],
        "action_std": acts.std(axis=0).tolist() if len(acts) else [],
        "action_histogram": {"vn": histogram(acts[:, 0]), "ve": histogram(acts[:, 1]),
                             "vd": histogram(acts[:, 2])} if len(acts) else {},
    }


def efficiency(bc, pid):
    return (1.0 - (bc - pid) / pid) * 100.0 if pid > 0 else float("nan")


def report_text(summary):
    b, p, c = summary["baseline"], summary["policy"], summary["comparison"]
    s = summary["settings"]
    L = ["=" * 78, "STAGE 1 STATISTICS REPORT",
         "PID Expert vs BC Policy - ArduPilot SITL + AirSim",
         f"Sensor mode: {summary['sensor_mode']} | Firmware: {summary['firmware']} | "
         f"Model trained: {summary['model_trained_at']}",
         f"Episodes: {s['episodes']} per controller x {s['steps']} steps at {s['hz']:.0f} Hz | "
         f"targets: alt {s['alt_min']}-{s['alt_max']} m, +/-{s['xy_offset']} m xy, +/-{s['z_offset']} m z (seed 42)",
         "=" * 78, "", "TABLE 1 - MAIN COMPARISON", "-" * 78,
         f"{'Metric':<40}{'PID Expert':>12}{'BC Policy':>12}{'Gap':>12}", "-" * 78]
    rows = [("Position error RMSE (m)", "pos_rmse_mean"), ("Altitude RMSE (m)", "alt_rmse_mean"),
            ("Altitude std within episode (m)", "alt_std_mean"),
            ("Mean horizontal distance (m)", "dist_mean"), ("Max horizontal distance (m)", "dist_max_mean"),
            ("End error, last 1 s (m)", "end_err_mean"),
            ("Success rate (%)", "success_rate"), ("Settled rate (%)", "settled_rate")]
    for label, k in rows:
        L.append(f"  {label:<38}{b[k]:>12.3f}{p[k]:>12.3f}{p[k]-b[k]:>+12.3f}")
    L += ["-" * 78,
          f"  {'Position RMSE ratio (BC / PID)':<38}{'':>12}{c['pos_rmse_ratio']:>12.3f}",
          f"  {'Transfer efficiency, position (%)':<38}{'':>12}{c['transfer_efficiency_pos']:>12.2f}",
          f"  {'Altitude RMSE ratio (BC / PID)':<38}{'':>12}{c['alt_rmse_ratio']:>12.3f}",
          f"  {'Transfer efficiency, altitude (%)':<38}{'':>12}{c['transfer_efficiency_alt']:>12.2f}",
          "", "TABLE 2 - PER EPISODE", "-" * 78,
          f"{'Ep':<4}{'Target alt':>11}{'PID pos RMSE':>14}{'BC pos RMSE':>13}"
          f"{'PID end':>10}{'BC end':>10}{'BC result':>14}", "-" * 78]
    n = min(b["n_episodes"], p["n_episodes"])
    for i in range(n):
        L.append(f"{i+1:<4}{b['target_alts'][i]:>11.2f}{b['ep_pos_rmse'][i]:>14.3f}"
                 f"{p['ep_pos_rmse'][i]:>13.3f}{b['ep_end_err'][i]:>10.3f}{p['ep_end_err'][i]:>10.3f}"
                 f"{p['ep_reason'][i]:>14}")
    L += ["", "ACTION DISTRIBUTION (m/s)", "-" * 78]
    for ch in ("vn", "ve", "vd"):
        hb, hp = b["action_histogram"][ch], p["action_histogram"][ch]
        L.append(f"  {ch}: PID mean {hb['mean']:+.4f} std {hb['std']:.4f} | "
                 f"BC mean {hp['mean']:+.4f} std {hp['std']:.4f}")
    L += ["", "DEFINITIONS", "-" * 78,
          f"  Success: no safety stop and end error < {SUCCESS_M} m (end error = mean of last 20 steps)",
          f"  Settled: end error < {SETTLED_M} m",
          "  Transfer efficiency = (1 - (BC_RMSE - PID_RMSE) / PID_RMSE) x 100%",
          f"  Real loop time: PID {b['timing']['dt_mean']*1000:.1f} ms, BC {p['timing']['dt_mean']*1000:.1f} ms",
          "=" * 78]
    return "\n".join(L)


def main(args):
    run_dir = run_folder(args.run)
    model_dir = run_dir / "models"
    model = HoverPolicy()
    model.load_state_dict(torch.load(model_dir / "hover_policy_best.pth", map_location="cpu"))
    model.eval()
    norm = np.load(model_dir / "obs_norm.npz")
    mean, std = norm["mean"], norm["std"]
    info_path = model_dir / "model_info.json"
    model_info = json.load(open(info_path)) if info_path.exists() else {}

    print("\n" + "=" * 70)
    print(f"STAGE 1 STATISTICS: PID EXPERT vs BC POLICY (ArduPilot) - {run_dir.name.upper()}")
    print("=" * 70)
    print(f"Run folder: {run_dir}")

    link = APLink(args.connect)
    link.wait_state()
    firmware = link.firmware_version()
    params = link.get_params(["EK3_SRC1_POSXY", "EK3_SRC1_VELXY"])
    mode = "optflow" if params.get("EK3_SRC1_VELXY") == 5 else \
        ("gps" if params.get("EK3_SRC1_POSXY") == 3 else "unknown")
    link.wait_state()
    if not link.armed:
        link.takeoff_nogps(0.5 * (args.alt_min + args.alt_max))
    else:
        link.set_mode("GUIDED")

    pid = PIDExpertHover(dt=1.0 / args.hz)

    def bc_act(obs):
        with torch.no_grad():
            a = model(torch.as_tensor(normalize(obs, mean, std)).unsqueeze(0)).squeeze(0).numpy()
        return np.clip(a, -ACT_LIMIT, ACT_LIMIT)

    def settle(seconds=10.0):
        # fly back to HOME between PID and BC, so both start from the same place
        start = np.array([home[0], home[1], -0.5 * (args.alt_min + args.alt_max)])
        rk = RateKeeper(args.hz)
        for _ in range(int(seconds * args.hz)):
            link.update()
            o = link.obs(start)
            link.send_velocity(*pid.get_action({"position": o[0:3], "velocity": o[3:6]}))
            rk.wait()
        link.hold()

    link.update()
    home = link.pos.copy()      # BOTH controllers get the same targets around this point
    print(f"Home: N {home[0]:.2f} E {home[1]:.2f} m | safety fence {args.xy_offset + FENCE_M:.1f} m")

    try:
        settle()          # PID starts settled at home too - same start as BC (fair comparison)
        pid_eps, pid_t = run_controller(
            link, "PART 1: PID Expert (baseline)",
            lambda o: pid.get_action({"position": o[0:3], "velocity": o[3:6]}), pid.reset, args, home)
        settle()
        bc_eps, bc_t = run_controller(link, "PART 2: BC Policy (neural network)", bc_act,
                                      lambda: None, args, home)
    finally:
        link.hold()
        link.land()                      # both controllers done (or stopped) - land

    base = build_stats("PID Expert", pid_eps, pid_t)
    pol = build_stats("BC Policy", bc_eps, bc_t)
    summary = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "sensor_mode": mode, "firmware": firmware, "settings": vars(args),
        "run": run_dir.name, "model_dir": str(model_dir), "model_trained_at": model_info.get("trained_at", "?"),
        "dataset": model_info.get("dataset", "?"),
        "baseline": base, "policy": pol,
        "comparison": {
            "pos_rmse_ratio": pol["pos_rmse_mean"] / base["pos_rmse_mean"],
            "alt_rmse_ratio": pol["alt_rmse_mean"] / base["alt_rmse_mean"],
            "transfer_efficiency_pos": efficiency(pol["pos_rmse_mean"], base["pos_rmse_mean"]),
            "transfer_efficiency_alt": efficiency(pol["alt_rmse_mean"], base["alt_rmse_mean"]),
        },
        "definitions": {"success": f"no safety stop and end error < {SUCCESS_M} m",
                        "settled": f"end error < {SETTLED_M} m",
                        "transfer_efficiency": "(1 - (BC_RMSE - PID_RMSE) / PID_RMSE) x 100%"},
    }

    out = run_dir / "results" / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(out / f"statistics_summary_{stamp}.pkl", "wb") as f:
        pickle.dump({**summary, "episodes": {"pid": pid_eps, "bc": bc_eps}}, f)
    with open(out / f"statistics_summary_{stamp}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    text = report_text(summary)
    with open(out / f"statistics_report_{stamp}.txt", "w") as f:
        f.write(text)

    print("\n" + text)
    print(f"\nSaved in {out}: statistics_summary_{stamp}.pkl / .json, statistics_report_{stamp}.txt")
    print("Next: python rmse_baseline.py | python vd_distribution.py | python make_stage1_summary.py")


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
    main(parser.parse_args())
