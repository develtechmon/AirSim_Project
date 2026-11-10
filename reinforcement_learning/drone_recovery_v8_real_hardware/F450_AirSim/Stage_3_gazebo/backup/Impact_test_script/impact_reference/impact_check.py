"""
IMPACT CHECK (Stage 3, ArduPilot SITL + AirSim)
===============================================
Same idea as Stage 2's wind_check.py: BEFORE any training, prove that the
disturbance works in SITL and measure what it does to the drone.

How the impact is made (same way in SITL and on the real drone):
    the drone hovers in GUIDED; for a short moment we send ArduPilot
    BODY-RATE commands (SET_ATTITUDE_TARGET, rates only) - "spin this hard
    for this long" - which throws the drone into a flip, spin or hit.
    Then the impact stops and a recovery controller takes over.
    Here the recovery controller is ArduPilot itself (GUIDED, hold still):
    that is the BASELINE the Stage 3 network must beat later.

Three disturbance types (as your original Stage 3), three levels (curriculum):
    flip   roll OR pitch rate 6 rad/s for 0.40 s        (tumble)
    spin   yaw rate 6 rad/s for 1.0 s                   (yaw spin)
    bird   roll 4 + pitch 4 + yaw 3 rad/s for 0.25 s    (short sharp hit)
    level  easy x0.7 | medium x1.0 | hard x1.5 (rates scaled)
    direction (+/-, roll or pitch) random from --seed, so a rerun is identical

Recovered (your original criterion) = all at once, held for 0.5 s:
    |body rate| < 1.2 rad/s, tilt < 30 deg (upright), height > 3 m, horizontal speed < 3 m/s

Hover height 20 m (a flip loses height; 2 m like Stage 1/2 would hit the ground).
Before every impact the drone flies back to the same start point and holds calmly for 3 s.
Safety: if height drops below 6 m the impact is cut at once and ArduPilot recovers.

Usage (in Stage_3, AirSim + SITL in GPS mode running):
    py .\\impact_check.py                                  (all 3 types x 3 levels, 1 each)
    py .\\impact_check.py --types flip --levels hard --repeats 3
Saves runs\\impact_check\\impact_check_<date>.csv (one line per impact)
      runs\\impact_check\\impact_check_<date>_steps.csv (every 50 Hz step, for plots)
      runs\\impact_check\\impact_check_<date>.json (settings + summary)
"""

import argparse
import csv
import datetime
import json
import time
from pathlib import Path

import numpy as np

from ap_link import APLink, RateKeeper

HZ = 50.0
ALT = 20.0                 # m hover height for impacts
ALT_CUT = 6.0              # m: below this the impact is stopped at once
HOLD_S = 3.0               # s calm hover before each impact
RECOVER_MAX_S = 10.0       # s: not recovered by then = FAIL
HELD_S = 0.5               # s the recovery criterion must hold
RATE_OK = 1.2              # rad/s   (original criterion)
TILT_OK = 30.0             # deg     upright
ALT_OK = 3.0               # m
HSPEED_OK = 3.0            # m/s
RETURN_MAX_S = 40.0        # s max to fly back to the start point between impacts
OUT = Path("runs") / "impact_check"

DISTURBANCES = {           # base body rates (rad/s) and duration (s)
    "flip": {"rates": (6.0, 0.0, 0.0), "dur": 0.40},
    "spin": {"rates": (0.0, 0.0, 6.0), "dur": 1.00},
    "bird": {"rates": (4.0, 4.0, 3.0), "dur": 0.25},
}
LEVELS = {"easy": 0.7, "medium": 1.0, "hard": 1.5}


def rates_for(kind, scale, rng):
    """Body rates for one impact: base x level, random sign / axis (seeded)."""
    p, q, r = (v * scale for v in DISTURBANCES[kind]["rates"])
    s = lambda: 1.0 if rng.rand() < 0.5 else -1.0
    if kind == "flip":
        if rng.rand() < 0.5:            # tumble about pitch instead of roll
            p, q = 0.0, p
        return p * s(), q * s(), r
    return p * s(), q * s(), r * s()


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
        v = np.clip(0.8 * err, -1.5, 1.5)
        link.send_velocity(v[0], v[1], float(np.clip(v[2], -1.0, 1.0)))
        rk.wait()
        calm = np.linalg.norm(err) < 0.5 and np.linalg.norm(link.vel) < 0.3 and link.tilt_deg < 5
        calm_since = (calm_since or time.time()) if calm else None
        if calm_since and time.time() - calm_since >= hold_s:
            return True
    print("   (did not get back to the start point calmly - continuing)")
    return False


def one_impact(link, kind, level, rng, n, rows):
    target = link.pos.copy()
    target[2] = -ALT
    fly_back(link, target, HOLD_S)
    p, q, r = (v + 0.0 for v in rates_for(kind, LEVELS[level], rng))
    dur = DISTURBANCES[kind]["dur"]
    alt0 = link.alt
    peak_tilt = peak_rate = 0.0
    cut = False
    rk = RateKeeper(HZ)
    t0 = time.time()
    # ---- impact: rate commands (thrust 0.5 = hold climb rate 0)
    while time.time() - t0 < dur:
        link.send_rates(p, q, r, 0.5)
        rk.wait()
        link.update()
        s = state(link, target)
        peak_tilt, peak_rate = max(peak_tilt, s["tilt"]), max(peak_rate, s["rate"])
        rows.append([n, kind, level, "impact", round(time.time() - t0, 3), round(s["tilt"], 1),
                     round(s["rate"], 2), round(s["alt"], 2), round(s["hspeed"], 2), round(s["dist"], 2)])
        if s["alt"] < ALT_CUT:
            cut = True
            break
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
        rows.append([n, kind, level, "recovery", round(time.time() - t0, 3), round(s["tilt"], 1),
                     round(s["rate"], 2), round(s["alt"], 2), round(s["hspeed"], 2), round(s["dist"], 2)])
        if s["ok"]:
            held_since = held_since or time.time()
            if time.time() - held_since >= HELD_S:
                rec_t = held_since - t_rel
                break
        else:
            held_since = None
    link.update()
    info = {"n": n, "type": kind, "level": level, "cmd_rates": [round(p, 2), round(q, 2), round(r, 2)],
            "cmd_duration_s": dur, "peak_tilt_deg": round(peak_tilt, 1), "peak_rate_rads": round(peak_rate, 2),
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
    print(f"Recovery (baseline): ArduPilot GUIDED hold | criterion: rate < {RATE_OK} rad/s, "
          f"tilt < {TILT_OK:.0f} deg, alt > {ALT_OK:.0f} m, h-speed < {HSPEED_OK:.0f} m/s for {HELD_S} s")
    print()
    link = APLink(args.conn, rate_hz=50)
    link.wait_state()
    link.update()
    if not link.armed:
        link.takeoff_nogps(ALT)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUT / f"impact_check_{stamp}"
    results, rows, n = [], [], 0
    print("=" * 70)
    try:
        for kind in args.types:
            for level in args.levels:
                for _ in range(args.repeats):
                    n += 1
                    e = one_impact(link, kind, level, rng, n, rows)
                    results.append(e)
                    rec = f"{e['recovery_time_s']:.2f}s" if e["recovered"] else "NOT RECOVERED"
                    print(f"Impact {n:2d} | {kind:<4} {level:<6} | rates {e['cmd_rates']} rad/s x {e['cmd_duration_s']}s | "
                          f"peak tilt {e['peak_tilt_deg']:5.1f}° | peak rate {e['peak_rate_rads']:4.1f} rad/s | "
                          f"recovered {rec} | height loss {e['alt_loss_m']:.1f} m"
                          + (" | CUT (too low)" if e["impact_cut_low"] else ""))
    except KeyboardInterrupt:
        print("\nStopped by user - saving what we have")
    finally:
        print("=" * 70)
        cols = ["n", "type", "level", "cmd_rates", "cmd_duration_s", "peak_tilt_deg", "peak_rate_rads",
                "recovered", "recovery_time_s", "alt_loss_m", "drift_m", "impact_cut_low"]
        with open(f"{base}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for e in results:
                w.writerow([e[c] for c in cols])
        with open(f"{base}_steps.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "type", "level", "phase", "t_s", "tilt_deg", "rate_rads", "alt_m", "hspeed_ms", "dist_m"])
            w.writerows(rows)
        summary = {}
        for kind in args.types:
            for level in args.levels:
                ee = [e for e in results if e["type"] == kind and e["level"] == level]
                if ee:
                    rt = [e["recovery_time_s"] for e in ee if e["recovered"]]
                    summary[f"{kind}/{level}"] = {
                        "impacts": len(ee), "recovered_pct": 100.0 * len(rt) / len(ee),
                        "mean_peak_tilt_deg": float(np.mean([e["peak_tilt_deg"] for e in ee])),
                        "mean_recovery_s": float(np.mean(rt)) if rt else None,
                        "mean_alt_loss_m": float(np.mean([e["alt_loss_m"] for e in ee]))}
        json.dump({"date": stamp, "settings": vars(args), "disturbances": DISTURBANCES, "levels": LEVELS,
                   "criterion": {"rate_rads": RATE_OK, "tilt_deg": TILT_OK, "alt_m": ALT_OK,
                                 "hspeed_ms": HSPEED_OK, "held_s": HELD_S, "max_s": RECOVER_MAX_S},
                   "summary": summary, "impacts": results}, open(f"{base}.json", "w"), indent=2)
        if summary:
            print("\n📊 SUMMARY (baseline = ArduPilot recovers by itself)")
            print(f"{'Type/level':<14}{'Recovered':>10}{'Peak tilt':>11}{'Recovery':>10}{'Height loss':>13}")
            for k, s in summary.items():
                rec = f"{s['mean_recovery_s']:.2f} s" if s["mean_recovery_s"] is not None else "-"
                print(f"{k:<14}{s['recovered_pct']:>9.0f}%{s['mean_peak_tilt_deg']:>10.0f}°{rec:>10}"
                      f"{s['mean_alt_loss_m']:>11.1f} m")
        print(f"\nSaved: {base}.csv / _steps.csv / .json")
        link.land()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["flip", "spin", "bird"], choices=list(DISTURBANCES))
    ap.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"], choices=list(LEVELS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    main(ap.parse_args())
