"""
IMPACT CHECK (Stage 3, ArduPilot SITL + AirSim)
===============================================
Same idea as Stage 2's wind_check.py: BEFORE any training, prove that the
disturbance works in SITL and measure what it does to the drone.

How the impact is made (same way in SITL and on the real drone):
    the drone hovers in GUIDED; then we send ArduPilot BODY-RATE commands
    (SET_ATTITUDE_TARGET, rates only) - "tumble this fast" - and keep them
    until the drone has turned a set number of times (counted from the gyro).
    Then the impact stops and a recovery controller takes over.
    Here the recovery controller is ArduPilot itself (GUIDED, hold still):
    that is the BASELINE the Stage 3 network must beat later.

Violent multi-turn impacts (levels = how many turns in the air):
    type   axis                         rate        easy     medium   hard
    flip   roll OR pitch (random)       12 rad/s    1 turn   2 turns  3 turns
    spin   yaw                          10 rad/s    1 turn   2 turns  3 turns
    bird   roll + pitch + yaw together  8/8/5 rad/s 0.5 turn 1 turn   1.5 turns
    1 turn = 360 deg. Direction (+/-, roll or pitch) random from --seed: a rerun is identical.
    Safety cap: an impact never lasts longer than 3x its planned time.

Impact mode: ArduPilot normally limits how fast it may START a rotation
(ATC_ACCEL_R_MAX / P_MAX / Y_MAX; on newer firmware ATC_ACC_R_MAX / P_MAX / Y_MAX), which made the
first version only tilt 100-157 deg. These three limits are set to 0 (= no limit) ONLY while the
impact is running and put back to your values the moment it ends - so the recovery (the baseline)
is flown by stock ArduPilot. They are also put back when the script ends (also on Ctrl+C).
The values are printed and saved. --keep-limits leaves them untouched.

Recovered (your original criterion) = all at once, held for 0.5 s:
    |body rate| < 1.2 rad/s, tilt < 30 deg (upright), height > 3 m, horizontal speed < 3 m/s

Hover height 60 m (first SITL run: a hard flip from 40 m lost 31 m, lowest point ~9 m).
Before every impact the drone flies back to the same start point and holds calmly for 3 s.
Safety: if height drops below 12 m the impact is cut at once and ArduPilot recovers.
"[fc] Potential Thrust Loss" during a flip is normal: a motor is at full power while tumbling.

Usage (in Stage_3, AirSim + SITL in GPS mode running):
    py .\\impact_check.py --types flip --levels hard          (quick test)
    py .\\impact_check.py                                      (all 3 types x 3 levels)
Saves runs\\impact_check\\impact_check_<date>.csv (one line per impact)
      runs\\impact_check\\impact_check_<date>_steps.csv (every 50 Hz step, for plots)
      runs\\impact_check\\impact_check_<date>.json (settings, limits used, summary)
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

HZ = 50.0
ALT = 60.0                 # m hover height for impacts (3 flips lost 31 m from 40 m in SITL)
ALT_CUT = 12.0             # m: below this the impact is stopped at once
HOLD_S = 3.0               # s calm hover before each impact
RECOVER_MAX_S = 15.0       # s: not recovered by then = FAIL
HELD_S = 0.5               # s the recovery criterion must hold
RATE_OK = 1.2              # rad/s   (original criterion)
TILT_OK = 30.0             # deg     upright
ALT_OK = 3.0               # m
HSPEED_OK = 3.0            # m/s
RETURN_MAX_S = 60.0        # s max to fly back to the start point between impacts
CAP = 3.0                  # impact never longer than CAP x planned time
OUT = Path("runs") / "impact_check"

DISTURBANCES = {           # body rates (rad/s) while tumbling
    "flip": (12.0, 0.0, 0.0),
    "spin": (0.0, 0.0, 10.0),
    "bird": (8.0, 8.0, 5.0),
}
TURNS = {                  # turns in the air per level
    "flip": {"easy": 1.0, "medium": 2.0, "hard": 3.0},
    "spin": {"easy": 1.0, "medium": 2.0, "hard": 3.0},
    "bird": {"easy": 0.5, "medium": 1.0, "hard": 1.5},
}
LEVELS = ["easy", "medium", "hard"]
LIMIT_NAMES = [["ATC_ACCEL_R_MAX", "ATC_ACCEL_P_MAX", "ATC_ACCEL_Y_MAX"],     # ArduPilot 4.4-4.6
               ["ATC_ACC_R_MAX", "ATC_ACC_P_MAX", "ATC_ACC_Y_MAX"]]           # newer firmware


def rates_for(kind, rng):
    """Body rates for one impact, random sign / axis (seeded)."""
    p, q, r = DISTURBANCES[kind]
    s = lambda: 1.0 if rng.rand() < 0.5 else -1.0
    if kind == "flip" and rng.rand() < 0.5:     # tumble about pitch instead of roll
        p, q = 0.0, p
    return p * s() + 0.0, q * s() + 0.0, r * s() + 0.0


def send_params(link, values):
    """Set ArduPilot parameters at once (no waiting - used right before / after an impact)."""
    for name, v in values.items():
        link.m.mav.param_set_send(link.m.target_system, link.m.target_component, name.encode(),
                                  float(v), 9)      # 9 = MAV_PARAM_TYPE_REAL32


def set_params(link, values, timeout=5):
    """Set ArduPilot parameters and read them back. Returns what the FC reports."""
    send_params(link, values)
    time.sleep(0.3)
    return link.get_params(list(values), timeout=timeout)


def read_limits(link):
    for names in LIMIT_NAMES:
        got = link.get_params(names)
        if len(got) == len(names):
            return got
    return {}


def state(link, target):
    tilt = link.tilt_deg
    rate = float(np.linalg.norm(link.rates))
    hspeed = float(np.linalg.norm(link.vel[0:2]))
    alt = link.alt
    ok = rate < RATE_OK and tilt < TILT_OK and alt > ALT_OK and hspeed < HSPEED_OK
    return {"tilt": tilt, "rate": rate, "hspeed": hspeed, "alt": alt,
            "dist": float(np.linalg.norm(link.pos - target)), "ok": ok}


def fly_back(link, target, hold_s, max_s=RETURN_MAX_S):
    """Fly back to the start point (gentle velocity commands), then hold there calmly for hold_s."""
    rk = RateKeeper(HZ)
    t0, calm_since = time.time(), None
    while time.time() - t0 < max_s:
        link.update()
        err = target - link.pos
        v = np.clip(0.8 * err, -2.0, 2.0)
        link.send_velocity(v[0], v[1], float(np.clip(v[2], -1.5, 1.5)))
        rk.wait()
        calm = np.linalg.norm(err) < 0.5 and np.linalg.norm(link.vel) < 0.3 and link.tilt_deg < 5
        calm_since = (calm_since or time.time()) if calm else None
        if calm_since and time.time() - calm_since >= hold_s:
            return True
    print("   (did not get back to the start point calmly - continuing)")
    return False


def log_row(rows, n, kind, level, phase, t, s, turns):
    rows.append([n, kind, level, phase, round(t, 3), round(s["tilt"], 1), round(s["rate"], 2),
                 round(turns, 2), round(s["alt"], 2), round(s["hspeed"], 2), round(s["dist"], 2)])


def one_impact(link, kind, level, rng, n, rows, target, limits):
    fly_back(link, target, HOLD_S)
    p, q, r = rates_for(kind, rng)
    goal = TURNS[kind][level]
    planned = goal * 2 * math.pi / float(np.linalg.norm([p, q, r]))
    alt0 = link.alt
    peak_tilt = peak_rate = turned = 0.0
    cut = False
    if limits:
        send_params(link, {k: 0 for k in limits})       # impact mode on
    rk = RateKeeper(HZ)
    t0 = t_prev = time.time()
    # ---- impact: tumble until the gyro has counted the planned turns
    while turned < goal and time.time() - t0 < CAP * planned:
        link.send_rates(p, q, r, 0.5)
        rk.wait()
        link.update()
        now = time.time()
        turned += float(np.linalg.norm(link.rates)) * (now - t_prev) / (2 * math.pi)
        t_prev = now
        s = state(link, target)
        peak_tilt, peak_rate = max(peak_tilt, s["tilt"]), max(peak_rate, s["rate"])
        log_row(rows, n, kind, level, "impact", now - t0, s, turned)
        if s["alt"] < ALT_CUT:
            cut = True
            break
    impact_s = time.time() - t0
    if limits:
        send_params(link, limits)                        # stock ArduPilot for the recovery
    t_rel = time.time()
    # ---- recovery: ArduPilot holds still (baseline)
    rec_t, held_since, min_alt = None, None, link.alt
    while time.time() - t_rel < RECOVER_MAX_S:
        link.send_velocity(0.0, 0.0, 0.0)
        rk.wait()
        link.update()
        s = state(link, target)
        peak_tilt, peak_rate = max(peak_tilt, s["tilt"]), max(peak_rate, s["rate"])
        min_alt = min(min_alt, s["alt"])
        log_row(rows, n, kind, level, "recovery", time.time() - t0, s, turned)
        if s["ok"]:
            held_since = held_since or time.time()
            if time.time() - held_since >= HELD_S:
                rec_t = held_since - t_rel
                break
        else:
            held_since = None
    link.update()
    info = {"n": n, "type": kind, "level": level, "cmd_rates": [round(p, 2), round(q, 2), round(r, 2)],
            "planned_turns": goal, "turns": round(turned, 2), "impact_s": round(impact_s, 2),
            "peak_tilt_deg": round(peak_tilt, 1), "peak_rate_rads": round(peak_rate, 2),
            "recovered": rec_t is not None, "recovery_time_s": round(rec_t, 2) if rec_t is not None else None,
            "alt_loss_m": round(alt0 - min_alt, 2), "drift_m": round(float(np.linalg.norm(link.pos - target)), 2),
            "impact_cut_low": cut}
    fly_back(link, target, 0.0)
    return info


def main(args):
    rng = np.random.RandomState(args.seed)
    print("\n" + "=" * 70)
    print("💥 STAGE 3 IMPACT CHECK - does the disturbance work in SITL?")
    print("=" * 70)
    print(f"Types: {', '.join(args.types)} | Levels: {', '.join(args.levels)} | repeats: {args.repeats}")
    for k in args.types:
        print(f"   {k:<5} rates {DISTURBANCES[k]} rad/s | turns " +
              ", ".join(f"{lv} {TURNS[k][lv]:g}" for lv in args.levels))
    print(f"Recovery (baseline): ArduPilot GUIDED hold | criterion: rate < {RATE_OK} rad/s, "
          f"tilt < {TILT_OK:.0f} deg, alt > {ALT_OK:.0f} m, h-speed < {HSPEED_OK:.0f} m/s for {HELD_S} s")
    print()
    link = APLink(args.conn, rate_hz=50)
    link.wait_state()
    original = read_limits(link)
    limits = {} if args.keep_limits else dict(original)
    if not args.keep_limits:
        if not original:
            raise SystemExit("Could not read the ATC_ACCEL/ATC_ACC limits - not changing anything "
                             "(run with --keep-limits to test without impact mode)")
        print(f"⚙️  Impact mode: {', '.join(f'{k} {v:g}' for k, v in original.items())} -> 0 (no limit) "
              "only during each impact")
        test = set_params(link, {k: 0 for k in original})
        ok = all(abs(v) < 1e-6 for v in test.values()) and len(test) == len(original)
        back = set_params(link, original)
        print(f"   check: set to 0 -> FC reports {test} | put back -> {back} | {'OK' if ok else 'FAILED'}")
        if not ok:
            raise SystemExit("The FC did not accept the limit change - stopping")
    link.update()
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUT / f"impact_check_{stamp}"
    results, rows, n = [], [], 0
    try:
        if not link.armed:
            link.takeoff_nogps(ALT)
        link.update()
        target = link.pos.copy()
        target[2] = -ALT
        print("=" * 70)
        for kind in args.types:
            for level in args.levels:
                for _ in range(args.repeats):
                    n += 1
                    e = one_impact(link, kind, level, rng, n, rows, target, limits)
                    results.append(e)
                    rec = f"{e['recovery_time_s']:.2f}s" if e["recovered"] else "NOT RECOVERED"
                    print(f"Impact {n:2d} | {kind:<4} {level:<6} | turns {e['turns']:.1f}/{e['planned_turns']:g} "
                          f"in {e['impact_s']:.2f}s | peak rate {e['peak_rate_rads']:4.1f} rad/s | "
                          f"peak tilt {e['peak_tilt_deg']:5.1f}° | recovered {rec} | height loss {e['alt_loss_m']:.1f} m"
                          + (" | CUT (too low)" if e["impact_cut_low"] else ""))
    except KeyboardInterrupt:
        print("\nStopped by user - saving what we have")
    finally:
        print("=" * 70)
        if limits:
            back = set_params(link, original)
            print(f"⚙️  Limits restored: {back}")
        cols = ["n", "type", "level", "cmd_rates", "planned_turns", "turns", "impact_s", "peak_rate_rads",
                "peak_tilt_deg", "recovered", "recovery_time_s", "alt_loss_m", "drift_m", "impact_cut_low"]
        with open(f"{base}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for e in results:
                w.writerow([e[c] for c in cols])
        with open(f"{base}_steps.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "type", "level", "phase", "t_s", "tilt_deg", "rate_rads", "turns", "alt_m",
                        "hspeed_ms", "dist_m"])
            w.writerows(rows)
        summary = {}
        for kind in args.types:
            for level in args.levels:
                ee = [e for e in results if e["type"] == kind and e["level"] == level]
                if ee:
                    rt = [e["recovery_time_s"] for e in ee if e["recovered"]]
                    summary[f"{kind}/{level}"] = {
                        "impacts": len(ee), "recovered_pct": 100.0 * len(rt) / len(ee),
                        "mean_turns": float(np.mean([e["turns"] for e in ee])),
                        "mean_peak_rate_rads": float(np.mean([e["peak_rate_rads"] for e in ee])),
                        "mean_recovery_s": float(np.mean(rt)) if rt else None,
                        "mean_alt_loss_m": float(np.mean([e["alt_loss_m"] for e in ee]))}
        json.dump({"date": stamp, "settings": vars(args), "disturbances_rads": DISTURBANCES, "turns": TURNS,
                   "limits_original": original, "impact_mode": bool(limits), "hover_alt_m": ALT,
                   "criterion": {"rate_rads": RATE_OK, "tilt_deg": TILT_OK, "alt_m": ALT_OK,
                                 "hspeed_ms": HSPEED_OK, "held_s": HELD_S, "max_s": RECOVER_MAX_S},
                   "summary": summary, "impacts": results}, open(f"{base}.json", "w"), indent=2)
        if summary:
            print("\n📊 SUMMARY (baseline = ArduPilot recovers by itself)")
            print(f"{'Type/level':<14}{'Recovered':>10}{'Turns':>8}{'Peak rate':>12}{'Recovery':>10}{'Height loss':>13}")
            for k, s in summary.items():
                rec = f"{s['mean_recovery_s']:.2f} s" if s["mean_recovery_s"] is not None else "-"
                print(f"{k:<14}{s['recovered_pct']:>9.0f}%{s['mean_turns']:>8.1f}{s['mean_peak_rate_rads']:>8.1f} rad/s"
                      f"{rec:>10}{s['mean_alt_loss_m']:>11.1f} m")
        print(f"\nSaved: {base}.csv / _steps.csv / .json")
        link.land(timeout=150)                         # from 60 m LAND takes about a minute


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["flip", "spin", "bird"], choices=list(DISTURBANCES))
    ap.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-limits", action="store_true", help="do not touch the ATC_ACCEL limits")
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    main(ap.parse_args())
