"""
REAL DRONE TEST: HOVER IN FAN WIND (Raspberry Pi Zero 2 + ESP32 ArduPilot + MTF-01)
==================================================================================
Runs ON THE PI. The same test as test_stage2_policy.py, but the "wind" is a house fan
and the drone is real. Same metric definitions as SITL and the paper, so the numbers
go side by side in the thesis.

Who flies:
    1. YOU take off in AltHold, then switch to PosHold and hover (about 1.5-2 m, sticks centred)
    2. flip the mode switch to GUIDED  -> the Pi flies: target = where the drone is now
    3. the Pi flies one episode (500 steps x 20 Hz = 25 s), then switches the drone
       back to PosHold itself and prints the result
    4. to fly the next episode: flip the switch away from GUIDED and back
    ANY TIME: flip the switch away from GUIDED -> you have control instantly
             (the Pi sees the mode change and stops sending at once)

The Pi also switches to PosHold by itself (and stops) if:
    > --fence m from the target (default 1.5 m, like Stage 1) or > 0.75 m in height,
    tilt > 35 deg, height < 0.4 m, or no position from the FC for 0.5 s.

Controllers (--model):
    warm    Stage 2 network (runs/run<N>/onnx/warm.onnx, from export_onnx.py)
    stage1  Stage 1 network (runs/run<N>/onnx/stage1.onnx) - the "before Stage 2" baseline
    pid     Stage 1 teacher (pid_expert.py)                - the reference

Same as SITL: observation (13, raw - the scaling is inside the .onnx), action = velocity
N/E/D, clipped to 2.0 m/s (xy) and 1.0 m/s (z) before sending, 20 Hz loop.

Heading: in training the drone always faced the same way (the heading is part of the
quaternion input). The Pi refuses to fly if the drone faces more than 20 deg away from
that heading - turn the drone (yaw) and engage again. --check prints the heading.

Indoor (MTF-01, no GPS): GUIDED needs an EKF origin. The script sends it when it connects
(--origin-lat/--origin-lon, default Kuching), so START THE SCRIPT BEFORE TAKEOFF.
With a GPS fix (outdoor backup) ArduPilot uses the GPS origin and ignores this.

Usage (on the Pi, in ~/Stage_2):
    python3 real_fan_test.py --check                         (link test: props OFF, no commands sent)
    python3 real_fan_test.py --model warm --fan off --episodes 3
    python3 real_fan_test.py --model warm --fan high --fan-dist 1.5 --episodes 3
Saves runs/run<N>/results/real_<model>_fan-<fan>_<date>.csv / .json / _steps.csv
"""

import argparse
import csv
import datetime
import json
import math
import time
from pathlib import Path

import numpy as np

from ap_link import APLink, RateKeeper

HZ = 20.0                  # same as SITL
VMAX_XY_SEND = 2.0         # m/s, same send limits as drone_hover_disturbance_env.py
VMAX_Z_SEND = 1.0          # m/s
STABLE_M = 0.5             # "held" = within this distance (same as SITL)
FENCE_Z = 0.75             # m, height error that stops the Pi
TILT_STOP = 35.0           # deg
ALT_MIN = 0.4              # m
ALT_ENGAGE = 0.8           # m, the Pi does not take over below this height
STALE_S = 0.5              # s without a position message -> stop
HEADING_TOL = 20.0         # deg from the training heading
RUNS = Path("runs")


def run_folder(run=None):
    nums = [int(p.name[3:]) for p in RUNS.glob("run*") if p.name[3:].isdigit()]
    if run is not None:
        return RUNS / f"run{int(run)}"
    if not nums:
        raise SystemExit("No runs/run<N> folder - copy Stage_2/runs/run1/onnx to the Pi first")
    return RUNS / f"run{max(nums)}"


def heading_deg(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def load_controller(name, run_dir):
    """Returns (act(obs_raw) -> velocity N/E/D, description, training heading or None)."""
    if name == "pid":
        from pid_expert import EXPERT_DESCRIPTION, PIDExpertHover
        teacher = PIDExpertHover()
        return (lambda o: teacher.get_action({"position": o[0:3], "velocity": o[3:6]}),
                f"Stage 1 teacher ({EXPERT_DESCRIPTION})", None)
    import onnxruntime as ort
    path = run_dir / "onnx" / f"{name}.onnx"
    if not path.exists():
        raise SystemExit(f"   ❌ {path} not found - run export_onnx.py on the PC and copy the onnx folder")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    yaw = float(json.loads(meta["training_heading_deg"]))

    def act(o):
        return sess.run(None, {"obs": o.reshape(1, 13).astype(np.float32)})[0][0]
    return act, str(path), yaw


def pos_age(link):
    m = link.m.messages.get("LOCAL_POSITION_NED")
    return time.time() - m._timestamp if m is not None else 1e9


def send_limited(link, a):
    a = np.asarray(a, dtype=np.float64)
    v = [float(np.clip(a[0], -VMAX_XY_SEND, VMAX_XY_SEND)), float(np.clip(a[1], -VMAX_XY_SEND, VMAX_XY_SEND)),
         float(np.clip(a[2], -VMAX_Z_SEND, VMAX_Z_SEND))]
    link.send_velocity(*v)
    return v


def to_poshold(link):
    link.hold()
    link.m.set_mode("POSHOLD")


def check_link(link, act, yaw_train, seconds=10):
    """Props OFF. Shows what the Pi receives and what the network WOULD send. Sends nothing."""
    print("\n🔌 LINK CHECK - props OFF, nothing is sent to the motors\n")
    p0, t0 = link.m.mav.total_packets_received, time.time()
    rk = RateKeeper(HZ)
    target = None
    for i in range(int(seconds * HZ)):
        link.update()
        if link.ready() and target is None:
            target = link.pos.copy() + np.array([0.3, 0.0, 0.0])     # pretend target 0.3 m north
        if link.ready() and i % int(HZ) == 0:
            o = link.obs(target)
            a = act(o)
            hd = heading_deg(link.q)
            hd_txt = (f"{hd:6.1f} deg (training {yaw_train:.0f}, "
                      f"{'OK' if abs(angle_diff(hd, yaw_train)) <= HEADING_TOL else 'TURN THE DRONE'})"
                      if yaw_train is not None else f"{hd:6.1f} deg")
            print(f"mode {link.mode:<9} armed {str(link.armed):<5} | height {link.alt:5.2f} m | "
                  f"vel {np.round(link.vel, 2)} | tilt {link.tilt_deg:4.1f} | heading {hd_txt} | "
                  f"pos age {pos_age(link) * 1000:4.0f} ms | would send N {a[0]:+.2f} E {a[1]:+.2f} D {a[2]:+.2f}")
        rk.wait()
    rate = (link.m.mav.total_packets_received - p0) / (time.time() - t0)
    s = rk.stats()
    print("\n" + "=" * 70)
    print(f"MAVLink packets: {rate:.0f}/s (expect > 100: position + attitude at 50 Hz each)")
    print(f"Pi loop: {1 / max(s['dt_mean'], 1e-6):.1f} Hz, worst step {s['dt_max'] * 1000:.0f} ms, "
          f"overruns {s['overruns']}")
    ok = link.ready() and rate > 100 and pos_age(link) < STALE_S
    print("✅ LINK OK - the 'would send' N value should be POSITIVE (target is 0.3 m north)" if ok
          else "❌ LINK NOT OK - check wiring, SERIAL2_BAUD, and that the EKF has a position (PosHold works?)")
    print("=" * 70 + "\n")


def finish(rec, reason, steps, rk):
    d, h, t = (np.array(rec[k]) if rec[k] else np.array([np.nan]) for k in ("dist", "hdist", "tilt"))
    s = rk.stats()
    return {"reason": reason, "success": reason == "timeout", "ep_steps": steps,
            "ep_mean_dist": float(np.nanmean(d)), "ep_max_dist": float(np.nanmax(d)),
            "ep_mean_hdist": float(np.nanmean(h)), "ep_max_hdist": float(np.nanmax(h)),
            "ep_end_dist": float(np.nanmean(d[-20:])), "ep_rmse": float(np.sqrt(np.nanmean(d ** 2))),
            "ep_hold_pct": float(100.0 * np.nanmean(d < STABLE_M)),
            "ep_mean_tilt": float(np.nanmean(t)), "ep_max_tilt": float(np.nanmax(t)),
            "loop_hz": 1.0 / max(s["dt_mean"], 1e-6), "loop_dt_max_ms": 1000 * s["dt_max"],
            "loop_overruns": s["overruns"]}


def fly_episode(link, act, yaw_train, args, n):
    """Wait for GUIDED, fly one episode, hand back to PosHold. Returns (info, step rows)."""
    print(f"\n⏳ Episode {n}: hover in PosHold (sticks centred), then flip the switch to GUIDED...")
    while True:
        link.update()
        if link.mode == "GUIDED" and link.armed and link.ready():
            hd = heading_deg(link.q)
            if link.alt < ALT_ENGAGE:
                print(f"   ⚠️  height {link.alt:.2f} m < {ALT_ENGAGE} m - climb higher, then engage again")
            elif yaw_train is not None and abs(angle_diff(hd, yaw_train)) > HEADING_TOL:
                print(f"   ⚠️  heading {hd:.0f} deg, network trained at {yaw_train:.0f} deg "
                      f"(+/-{HEADING_TOL:.0f}) - yaw the drone, then engage again")
            else:
                break
            to_poshold(link)
            while link.mode == "GUIDED":
                link.update()
                time.sleep(0.05)
        time.sleep(0.02)

    target = link.pos.copy()
    print(f"   ▶️  Pi flying | target N {target[0]:.2f} E {target[1]:.2f} height {-target[2]:.2f} m")
    rec = {"dist": [], "hdist": [], "tilt": []}
    rows = []
    rk = RateKeeper(HZ)
    reason, steps, t0 = None, 0, time.time()
    obs = link.obs(target)
    while True:
        sent = send_limited(link, act(obs))
        rk.wait()
        link.update()
        steps += 1
        obs = link.obs(target)
        dist, hdist = float(np.linalg.norm(obs[0:3])), float(np.linalg.norm(obs[0:2]))
        tilt = link.tilt_deg
        rec["dist"].append(dist)
        rec["hdist"].append(hdist)
        rec["tilt"].append(tilt)
        rows.append([n, steps, round(time.time() - t0, 3), *np.round(obs[0:6], 3), round(tilt, 2),
                     *np.round(sent, 3), link.mode])
        if link.mode != "GUIDED":
            reason = "pilot_takeover"
        elif pos_age(link) > STALE_S:
            reason = "link_lost"
        elif hdist > args.fence or abs(obs[2]) > FENCE_Z:
            reason = "fence"
        elif tilt > TILT_STOP:
            reason = "tilt"
        elif link.alt < ALT_MIN:
            reason = "too_low"
        elif steps >= args.max_steps:
            reason = "timeout"
        if reason:
            break
    if reason != "pilot_takeover":
        to_poshold(link)
    info = finish(rec, reason, steps, rk)
    info.update({"episode": n, "target_ned": [round(float(x), 3) for x in target],
                 "heading_deg": round(heading_deg(link.q), 1)})
    print(f"   ⏹️  {'Done' if reason == 'timeout' else 'Stopped: ' + reason} - drone in PosHold. "
          f"Flip the switch away from GUIDED.")
    return info, rows


def summarize(eps):
    if not eps:
        return {}
    g = lambda k: [e[k] for e in eps]
    succ = [e for e in eps if e["success"]]
    return {"episodes": len(eps), "successes": len(succ), "success_pct": 100.0 * len(succ) / len(eps),
            "avg_distance_m": float(np.mean([e["ep_mean_hdist"] for e in succ])) if succ else None,
            "rmse_m": float(np.mean(g("ep_rmse"))), "mean_dist_m": float(np.mean(g("ep_mean_hdist"))),
            "end_dist_m": float(np.mean(g("ep_end_dist"))), "max_dist_m": float(np.mean(g("ep_max_hdist"))),
            "hold_pct": float(np.mean(g("ep_hold_pct"))), "mean_tilt_deg": float(np.mean(g("ep_mean_tilt"))),
            "max_tilt_deg": float(np.max(g("ep_max_tilt"))), "loop_hz": float(np.mean(g("loop_hz"))),
            "avg_length_steps": float(np.mean(g("ep_steps"))),
            "fail_reasons": {r: g("reason").count(r) for r in set(g("reason")) if r != "timeout"}}


def main(args):
    run_dir = run_folder(args.run)
    print("\n" + "=" * 70)
    print("🧪 TESTING STAGE 2 ON THE REAL DRONE (13 OBSERVATIONS)")
    print("=" * 70)
    print(f"Fan: {args.fan}" + (f" at {args.fan_dist} m" if args.fan_dist else ""))
    print()
    act, desc, yaw_train = load_controller(args.model, run_dir)
    print(f"[1/3] Loading model: {desc}")
    print("   ✅ Model loaded" + (f" | training heading {yaw_train:.0f} deg" if yaw_train is not None else ""))
    print(f"\n[2/3] Connecting to the flight controller: {args.conn} @ {args.baud}")
    print(f"   EKF origin (only if no GPS fix; sets compass declination): "
          f"lat {args.origin_lat}, lon {args.origin_lon}")
    link = APLink(args.conn, baud=args.baud, rate_hz=50, origin=(args.origin_lat, args.origin_lon, 0.0))
    link.wait_state()
    print("   ✅ Position and attitude received")
    if args.check:
        check_link(link, act, yaw_train)
        return

    print(f"\n[3/3] Running {args.episodes} episodes ({args.max_steps} steps = {args.max_steps / HZ:.0f} s each)")
    print(f"   Pi stops and switches to PosHold at: > {args.fence} m, tilt > {TILT_STOP:.0f} deg, "
          f"height < {ALT_MIN} m, link lost")
    print("   YOU can take over at any time: flip the switch away from GUIDED")
    print("=" * 70)
    out = run_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out / f"real_{args.model}_fan-{args.fan}_{stamp}"
    eps, rows = [], []
    try:
        for i in range(args.episodes):
            e, r = fly_episode(link, act, yaw_train, args, i + 1)
            eps.append(e)
            rows += r
            print(f"Episode {i + 1:2d}/{args.episodes} | Steps: {e['ep_steps']:3d} | "
                  f"Success: {'✅' if e['success'] else '❌'} | Dist: {e['ep_mean_hdist']:.2f}m | "
                  f"Tilt: {e['ep_mean_tilt']:.1f}° (max: {e['ep_max_tilt']:.1f}) | Reason: {e['reason']}")
    except KeyboardInterrupt:
        to_poshold(link)
        print("\nStopped by user (Ctrl+C) - drone switched to PosHold, saving what we have")
    print("=" * 70)

    s = summarize(eps)
    if s:
        print("\n" + "=" * 70)
        print("📊 TEST RESULTS (real drone)")
        print("=" * 70)
        print(f"Success Rate: {s['success_pct']:.0f}% ({s['successes']}/{s['episodes']} episodes)")
        if s["successes"]:
            print(f"Average Distance: {s['avg_distance_m']:.2f}m (successful episodes)")
        print(f"RMSE: {s['rmse_m']:.3f} m | End offset: {s['end_dist_m']:.3f} m | "
              f"Held within {STABLE_M} m: {s['hold_pct']:.0f}%")
        print(f"Lean (fan strength): mean {s['mean_tilt_deg']:.1f}°, max {s['max_tilt_deg']:.1f}°")
        print(f"Pi loop: {s['loop_hz']:.1f} Hz (target {HZ:.0f})")
        if s["fail_reasons"]:
            print("Stop Reasons:")
            for r, n in s["fail_reasons"].items():
                print(f"   - {r}: {n}")
        print("=" * 70 + "\n")

    cols = ["episode", "reason", "success", "ep_steps", "ep_mean_hdist", "ep_max_hdist", "ep_rmse",
            "ep_mean_dist", "ep_end_dist", "ep_hold_pct", "ep_mean_tilt", "ep_max_tilt", "heading_deg",
            "loop_hz", "loop_dt_max_ms", "loop_overruns"]
    with open(f"{base}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in eps:
            w.writerow([round(e[c], 3) if isinstance(e[c], float) else e[c] for c in cols])
    with open(f"{base}_steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "step", "t_s", "err_n", "err_e", "err_d", "vel_n", "vel_e", "vel_d",
                    "tilt_deg", "cmd_n", "cmd_e", "cmd_d", "mode"])
        w.writerows(rows)
    json.dump({"model": args.model, "controller": desc, "fan": args.fan, "fan_dist_m": args.fan_dist,
               "notes": args.notes, "settings": vars(args), "training_heading_deg": yaw_train,
               "summary": s, "episodes": eps}, open(f"{base}.json", "w"), indent=2, default=str)
    print(f"Saved: {base}.csv / .json / _steps.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="warm", choices=["warm", "stage1", "pid"])
    ap.add_argument("--fan", default="off", help="label for the results: off / low / mid / high")
    ap.add_argument("--fan-dist", type=float, default=None, help="fan distance from the drone (m), label only")
    ap.add_argument("--notes", default="", help="free text saved with the results")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--fence", type=float, default=1.5, help="m from the target before the Pi gives up")
    ap.add_argument("--conn", default="/dev/serial0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--run", default=None)
    ap.add_argument("--check", action="store_true", help="link test only (props off), sends nothing")
    ap.add_argument("--origin-lat", type=float, default=1.5535,
                    help="your flying site latitude (default Kuching); ignored when GPS has a fix")
    ap.add_argument("--origin-lon", type=float, default=110.3593,
                    help="your flying site longitude (default Kuching)")
    main(ap.parse_args())
