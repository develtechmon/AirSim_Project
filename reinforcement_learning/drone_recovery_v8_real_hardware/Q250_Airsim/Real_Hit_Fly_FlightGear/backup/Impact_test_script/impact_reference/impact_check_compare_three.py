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

Hover height 30 m (note: ArduPilot fell to 3.1 m after a hard 3-turn flip from here). VIRTUAL GROUND at 5 m: going below it = FAILED (as if it hit the ground),
even if it recovers afterwards - so the real floor is never reached and SITL never needs a restart.
Impact cut: the tumble is stopped early if the drone could no longer stop above the virtual ground
(height minus braking distance at its falling speed < 5 m + 2 m margin) - then ArduPilot recovers.
Real crash (disarmed in the air, or on the floor): the script saves, stops and says so.
Before every impact the drone flies back to the same start point and holds calmly for 3 s.
"[fc] Potential Thrust Loss" / "Vibration compensation" / "GPS Glitch" during a flip are normal: a motor is at full power while tumbling.

--compare-recovery: every impact is flown 3 times (same hit, same seed, same start point);
only what happens AFTER the tumble changes:
    ardupilot    Test 1: ArduPilot alone ("hold still")                      = the baseline to beat
    limit-on     Test 2: "brake hard + climb 5 m/s", brake limit ON (stock)  = best PPO could do, current port
    limit-off    Test 3: "brake hard + climb 5 m/s", brake limit OFF         = braking only, limit removed
    spin-stop-fast  Test 4: like ardupilot, but the rotation limits (ATC_ACCEL) stay OFF after the hit
                 until the spin has stopped (< 3 rad/s)                      = can ArduPilot stop the spin faster?
    rate-rule    Test 5: a simple hand-made RATE + THRUST recovery (what PPO would control in Route B):
                 turn upright with body-rate commands, low thrust while upside down, full thrust to brake
                 once upright; rotation limits OFF and GUID_OPTIONS "thrust as thrust" ONLY until calm,
                 then ArduPilot holds position                               = what a rate-level recovery can do
Rescue: below the 5 m virtual ground a test is already FAILED; the script then saves the drone itself
(rotation limits off + full-thrust rate-rule) so a bad fall no longer crashes SITL.
--modes picks which tests to fly (default with --compare-recovery: all five).
    (loosened only from the hit until recovered: WPNAV_ACCEL_Z 500, WPNAV_SPEED_UP 500, PSC_JERK_Z 50 -
     ArduPilot's highest allowed values; GUIDED reads them when velocity control restarts after the hit)

Usage (in Stage_3, AirSim + SITL in GPS mode running):
    py .\\impact_check.py --types flip --levels hard          (quick test)
    py .\\impact_check.py                                      (all 3 types x 3 levels)
    py .\\impact_check.py --types flip --levels hard --repeats 3 --compare-recovery
    
    How to run the script
    py .\impact_check.py --types flip --levels hard --repeats 3 --compare-recovery --modes ardupilot spin-stop-fast rate-rule

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
ALT = 30.0                 # m hover height (note: a hard flip bottomed out at 3.1 m from here)
FLOOR = 5.0                # m virtual ground: below this = failed
MARGIN = 2.0               # m extra margin for the impact cut
BRAKE = 8.0                # m/s^2 braking ArduPilot can do (2 g thrust - 1 g), for the braking distance
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
    "flip": (16.0, 0.0, 0.0),       # SITL quad tops out ~14.4 rad/s; 16 = "as fast as it can"
    "spin": (0.0, 0.0, 10.0),
    "bird": (10.0, 10.0, 6.0),
}
TURNS = {                  # turns in the air per level
    "flip": {"easy": 1.0, "medium": 2.0, "hard": 3.0},
    "spin": {"easy": 1.0, "medium": 2.0, "hard": 3.0},
    "bird": {"easy": 0.5, "medium": 1.0, "hard": 1.5},
}
LEVELS = ["easy", "medium", "hard"]
RECOVERIES = ["ardupilot", "limit-on", "limit-off", "spin-stop-fast", "rate-rule"]
SPIN_STOPPED = 3.0         # rad/s: spin-stop-fast keeps the rotation limits off until below this
RULE_K = 6.0               # rate-rule: rad/s of turn rate per rad of tilt error
RULE_MAX = 15.0            # rad/s max turn rate it asks for
RULE_THR_MIN = 0.15        # thrust while still upside down / tilted > 60 deg
RULE_BRAKE = 0.12          # extra thrust per m/s of falling speed once upright (x hover thrust)
RULE_FULL_BELOW = 2.0      # m/s: upright and falling faster than this = full thrust
RESCUE_FLOOR = 3.0         # m: rescue as soon as the drone would reach this height ...
RESCUE_BRAKE = 6.0         # m/s^2 ... braking at this rate (your SITL quad: hover thrust 0.56 -> ~7-8 max)
# RESCUE: when a test goes below the 5 m virtual ground - or is falling so fast that even braking at
# 6 m/s^2 it would reach 3 m (so it will certainly pass 5 m) - it is FAILED; the script then rescues the
# drone itself (rotation limits off + rate-rule with full thrust) EARLY ENOUGH that SITL does not crash.
GUID_THRUST_BIT = 8        # GUID_OPTIONS bit 3: SET_ATTITUDE_TARGET thrust = real thrust (not climb rate)
CLIMB = 5.0                # m/s "brake hard + climb" command for limit-on / limit-off (your AirSim action limit)
FREE_NAMES = [["WPNAV_ACCEL_Z", "WPNAV_SPEED_UP", "PSC_JERK_Z"],        # ArduPilot 4.4-4.6
              ["WPNAV_ACCEL_Z", "WPNAV_SPEED_UP", "PSC_JERK_D"]]        # newer firmware
FREE_VALUES = {"WPNAV_ACCEL_Z": 500, "WPNAV_SPEED_UP": 500, "PSC_JERK_Z": 50, "PSC_JERK_D": 50}
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


def read_free(link):
    for names in FREE_NAMES:
        got = link.get_params(names)
        if len(got) == len(names):
            return got
    return {}


def rate_rule(link, hover, full=False):
    """Simple rate-level recovery: body rates that turn the drone upright + scheduled thrust."""
    w, x, y, z = link.q
    d = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])   # world-down in body
    axis = np.array([-d[1], d[0], 0.0])                     # body-down x world-down
    sn = float(np.linalg.norm(axis))
    ang = math.atan2(sn, d[2])                              # tilt error, 0..pi
    u = axis / sn if sn > 1e-3 else (np.array([1.0, 0.0, 0.0]) if d[2] < 0 else np.zeros(3))
    wr = min(RULE_K * ang, RULE_MAX) * u
    v_down = max(0.0, float(link.vel[2]))
    if d[2] < 0.5:
        thr = RULE_THR_MIN                                   # still tilted / upside down: do not push
    elif full or v_down > RULE_FULL_BELOW:
        thr = 1.0                                            # upright and still falling: full brake
    else:
        thr = float(np.clip(hover * (1 + RULE_BRAKE * v_down), hover, 1.0))
    return float(wr[0]), float(wr[1]), 0.0, thr


def brake_floor(link):
    """Lowest height the drone would reach if it started braking now."""
    v_down = max(0.0, float(link.vel[2]))
    return link.alt - v_down ** 2 / (2 * BRAKE)


def crashed(link):
    return (not link.armed) or link.alt < 0.3


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


def log_row(rows, n, kind, level, phase, t, s, turns, recovery="ardupilot"):
    rows.append([n, kind, level, phase, round(t, 3), round(s["tilt"], 1), round(s["rate"], 2),
                 round(turns, 2), round(s["alt"], 2), round(s["hspeed"], 2), round(s["dist"], 2), recovery])


def one_impact(link, kind, level, rates, n, rows, target, limits, recovery="ardupilot", free=None,
               guid=None, hover=0.4):
    fly_back(link, target, HOLD_S)
    p, q, r = rates
    goal = TURNS[kind][level]
    planned = goal * 2 * math.pi / float(np.linalg.norm([p, q, r]))
    alt0 = link.alt
    peak_tilt = peak_rate = turned = 0.0
    cut = False
    if limits:
        send_params(link, {k: 0 for k in limits})       # impact mode on
    if recovery == "limit-off":
        send_params(link, {k: FREE_VALUES[k] for k in free})   # loosen BEFORE velocity control restarts
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
        log_row(rows, n, kind, level, "impact", now - t0, s, turned, recovery)
        if brake_floor(link) < FLOOR + MARGIN:
            cut = True
            break
        if crashed(link):
            raise SystemExit("💥 REAL CRASH during the impact (disarmed or on the floor) - "
                             "results saved; restart SITL")
    impact_s = time.time() - t0
    keep_rot = recovery in ("spin-stop-fast", "rate-rule") and bool(limits)
    if limits and not keep_rot:
        send_params(link, limits)                        # stock ArduPilot for the recovery
    rule_on = recovery == "rate-rule"
    if rule_on:
        send_params(link, {"GUID_OPTIONS": int(guid) | GUID_THRUST_BIT})   # thrust field = real thrust
    spin_stop_t = rule_end_t = None
    rescue = False
    t_rel = time.time()
    # ---- recovery: ardupilot = hold still (baseline); limit-on / limit-off = brake + climb hard
    rec_t, held_since, min_alt, ground = None, None, link.alt, False
    alt_end_impact = link.alt
    while time.time() - t_rel < RECOVER_MAX_S:
        if rule_on:
            link.send_rates(*rate_rule(link, hover, full=rescue))
        elif recovery in ("ardupilot", "spin-stop-fast") or link.alt >= ALT:
            link.send_velocity(0.0, 0.0, 0.0)
        else:
            link.send_velocity(0.0, 0.0, -CLIMB)          # NED: negative = up
        rk.wait()
        link.update()
        s = state(link, target)
        peak_tilt, peak_rate = max(peak_tilt, s["tilt"]), max(peak_rate, s["rate"])
        min_alt = min(min_alt, s["alt"])
        ground = ground or s["alt"] < FLOOR
        doomed = s["alt"] - max(0.0, float(link.vel[2])) ** 2 / (2 * RESCUE_BRAKE) < RESCUE_FLOOR
        if doomed and not rescue:
            ground = True                                    # would pass 5 m even braking hard = FAILED
        if ground and not rescue and guid is not None and limits:
            rescue = rule_on = keep_rot = True               # already FAILED - now save the drone
            send_params(link, {**{k: 0 for k in limits}, "GUID_OPTIONS": int(guid) | GUID_THRUST_BIT})
        if crashed(link):
            raise SystemExit("💥 REAL CRASH during recovery (disarmed or on the floor) - "
                             "results saved; restart SITL")
        log_row(rows, n, kind, level, "recovery", time.time() - t0, s, turned, recovery)
        if spin_stop_t is None and s["rate"] < SPIN_STOPPED:
            spin_stop_t = time.time() - t_rel
        if rule_on and s["rate"] < RATE_OK and s["tilt"] < TILT_OK and float(link.vel[2]) < 1.0:
            rule_on = False                              # calm: hand back to ArduPilot (velocity hold)
            rule_end_t = rule_end_t if rescue else time.time() - t_rel
            send_params(link, {"GUID_OPTIONS": int(guid)})
        if keep_rot and ((recovery == "spin-stop-fast" and s["rate"] < SPIN_STOPPED) or
                         ((recovery == "rate-rule" or rescue) and not rule_on)):
            send_params(link, limits)                    # stock rotation limits again
            keep_rot = False
        if s["ok"]:
            held_since = held_since or time.time()
            if time.time() - held_since >= HELD_S:
                rec_t = held_since - t_rel
                break
        else:
            held_since = None
    if recovery == "limit-off":
        send_params(link, free)                          # stock limits again
    if keep_rot:
        send_params(link, limits)
    if rule_on:
        send_params(link, {"GUID_OPTIONS": int(guid)})
    link.update()
    info = {"n": n, "type": kind, "level": level, "recovery": recovery, "cmd_rates": [round(p, 2), round(q, 2), round(r, 2)],
            "planned_turns": goal, "turns": round(turned, 2), "impact_s": round(impact_s, 2),
            "peak_tilt_deg": round(peak_tilt, 1), "peak_rate_rads": round(peak_rate, 2),
            "recovered": bool(rec_t is not None and not ground),
            "recovery_time_s": round(rec_t, 2) if rec_t is not None else None,
            "alt_loss_m": round(alt0 - min_alt, 2), "min_alt_m": round(min_alt, 2), "virtual_ground": bool(ground),
            "drift_m": round(float(np.linalg.norm(link.pos - target)), 2), "impact_cut_low": bool(cut),
            "spin_stopped_s": round(spin_stop_t, 2) if spin_stop_t is not None else None,
            "alt_end_impact_m": round(alt_end_impact, 2),
            "rule_handover_s": round(rule_end_t, 2) if rule_end_t is not None else None,
            "rescued": bool(rescue)}
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
    modes = (args.modes or RECOVERIES) if args.compare_recovery else ["ardupilot"]
    if args.compare_recovery:
        print(f"Compare recovery: every impact flown {len(modes)}x - {', '.join(modes)} (same hit each time)")
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
    free, guid, hover = {}, None, 0.4
    got = link.get_params(["GUID_OPTIONS", "MOT_THST_HOVER"])
    if "GUID_OPTIONS" in got:
        guid = int(got["GUID_OPTIONS"])
        hover = float(got.get("MOT_THST_HOVER", 0.4))
    if "rate-rule" in modes and (guid is None or not limits):
        raise SystemExit("rate-rule needs GUID_OPTIONS readable and impact mode (no --keep-limits)")
    if guid is not None and limits:
        print(f"⚙️  rate-rule / rescue: GUID_OPTIONS {guid} -> {guid | GUID_THRUST_BIT} (thrust as thrust) only "
              f"until calm | hover thrust {hover:.2f}")
        print(f"🛟 Rescue ON: a test that goes below {FLOOR:g} m (or falls too fast to stop above it) is recorded "
              "as FAILED, then the script saves the drone early (no more SITL restarts)")
    else:
        print("⚠️  Rescue OFF (GUID_OPTIONS not readable or --keep-limits) - a bad fall can crash SITL")
    if "limit-off" in modes:
        free = read_free(link)
        if not free:
            raise SystemExit("Could not read WPNAV_ACCEL_Z / WPNAV_SPEED_UP / PSC_JERK_Z - cannot run the limit-off test")
        test = set_params(link, {k: FREE_VALUES[k] for k in free})
        back = set_params(link, free)
        ok = len(test) == len(free) and all(abs(test[k] - FREE_VALUES[k]) < 1e-3 for k in free)
        print(f"⚙️  Brake limit OFF (limit-off test): {', '.join(f'{k} {v:g}' for k, v in free.items())} -> "
              f"{', '.join(f'{k} {FREE_VALUES[k]:g}' for k in free)} (only from the hit until recovered)")
        print(f"   check: loosened -> FC reports {test} | put back -> {back} | {'OK' if ok else 'FAILED'}")
        if not ok:
            raise SystemExit("The FC did not accept brake limit OFF - stopping")
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
                    rates = rates_for(kind, rng)            # the same hit for every recovery mode
                    for mode in modes:
                        n += 1
                        e = one_impact(link, kind, level, rates, n, rows, target, limits, mode, free, guid, hover)
                        results.append(e)
                        spin = (f" | spin stopped {e['spin_stopped_s']:.2f}s after hit"
                                if e["spin_stopped_s"] is not None else " | spin not stopped")
                        rec = (f"recovered {e['recovery_time_s']:.2f}s" if e["recovered"] else
                               "FAILED - hit virtual ground" if e["virtual_ground"] else "FAILED - not recovered")
                        tag = f" | {mode:<14}" if args.compare_recovery else ""
                        print(f"Impact {n:2d} | {kind:<4} {level:<6}{tag} | turns {e['turns']:.1f}/{e['planned_turns']:g} "
                              f"in {e['impact_s']:.2f}s | peak rate {e['peak_rate_rads']:4.1f} rad/s | "
                              f"height loss {e['alt_loss_m']:4.1f} m (lowest {e['min_alt_m']:4.1f} m) | {rec}"
                              + spin + (" | impact cut early (braking)" if e["impact_cut_low"] else "")
                              + (" | rescued by the script below 5 m" if e["rescued"] else ""))
    except KeyboardInterrupt:
        print("\nStopped by user - saving what we have")
    finally:
        print("=" * 70)
        if limits:
            back = set_params(link, original)
            print(f"⚙️  Limits restored: {back}")
        if free:
            back = set_params(link, free)
            print(f"⚙️  Brake limit back ON (restored): {back}")
        if guid is not None and limits:
            back = set_params(link, {"GUID_OPTIONS": guid})
            print(f"⚙️  GUID_OPTIONS restored: {back}")
        cols = ["n", "type", "level", "recovery", "cmd_rates", "planned_turns", "turns", "impact_s", "peak_rate_rads",
                "peak_tilt_deg", "recovered", "recovery_time_s", "alt_loss_m", "min_alt_m", "virtual_ground", "drift_m",
                "impact_cut_low", "alt_end_impact_m", "spin_stopped_s", "rule_handover_s", "rescued"]
        with open(f"{base}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for e in results:
                w.writerow([e[c] for c in cols])
        with open(f"{base}_steps.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "type", "level", "phase", "t_s", "tilt_deg", "rate_rads", "turns", "alt_m",
                        "hspeed_ms", "dist_m", "recovery"])
            w.writerows(rows)
        summary = {}
        for kind in args.types:
            for level in args.levels:
              for mode in modes:
                ee = [e for e in results if e["type"] == kind and e["level"] == level and e["recovery"] == mode]
                if ee:
                    rt = [e["recovery_time_s"] for e in ee if e["recovered"]]
                    key = f"{kind}/{level}/{mode}" if args.compare_recovery else f"{kind}/{level}"
                    summary[key] = {
                        "below_5m": int(sum(e["virtual_ground"] for e in ee)),
                        "mean_spin_stop_s": (float(np.mean([e["spin_stopped_s"] for e in ee if e["spin_stopped_s"] is not None]))
                                             if any(e["spin_stopped_s"] is not None for e in ee) else None),
                        "mean_lowest_m": float(np.mean([e["min_alt_m"] for e in ee])),
                        "impacts": len(ee), "recovered_pct": 100.0 * len(rt) / len(ee),
                        "mean_turns": float(np.mean([e["turns"] for e in ee])),
                        "mean_peak_rate_rads": float(np.mean([e["peak_rate_rads"] for e in ee])),
                        "mean_recovery_s": float(np.mean(rt)) if rt else None,
                        "mean_alt_loss_m": float(np.mean([e["alt_loss_m"] for e in ee])),
                        "lowest_m": float(np.min([e["min_alt_m"] for e in ee]))}
        json.dump({"date": stamp, "settings": vars(args), "disturbances_rads": DISTURBANCES, "turns": TURNS,
                   "limits_original": original, "limit_off_original": free,
                   "limit_off_values": {k: FREE_VALUES[k] for k in free}, "climb_mps": CLIMB, "impact_mode": bool(limits), "hover_alt_m": ALT,
                   "virtual_ground_m": FLOOR, "cut_margin_m": MARGIN, "brake_mss": BRAKE,
                   "criterion": {"rate_rads": RATE_OK, "tilt_deg": TILT_OK, "alt_m": ALT_OK,
                                 "hspeed_ms": HSPEED_OK, "held_s": HELD_S, "max_s": RECOVER_MAX_S},
                   "summary": summary, "impacts": results}, open(f"{base}.json", "w"), indent=2, default=str)
        if summary:
            w0 = 30 if args.compare_recovery else 14
            print("\n📊 SUMMARY" + (" - same hits, recoveries compared" if args.compare_recovery
                                    else " (baseline = ArduPilot recovers by itself)"))
            print(f"{'Type/level' + ('/recovery' if args.compare_recovery else ''):<{w0}}{'Recovered':>10}{'Turns':>8}"
                  f"{'Peak rate':>12}{'Recovery':>10}{'Height loss':>13}{'Lowest':>9}{'Avg low':>9}"
                  f"{'Spin stop':>11}{'Below 5 m':>11}")
            for k, s in summary.items():
                rec = f"{s['mean_recovery_s']:.2f} s" if s["mean_recovery_s"] is not None else "-"
                sst = f"{s['mean_spin_stop_s']:.2f} s" if s["mean_spin_stop_s"] is not None else "-"
                print(f"{k:<{w0}}{s['recovered_pct']:>9.0f}%{s['mean_turns']:>8.1f}{s['mean_peak_rate_rads']:>8.1f} rad/s"
                      f"{rec:>10}{s['mean_alt_loss_m']:>11.1f} m{s['lowest_m']:>7.1f} m{s['mean_lowest_m']:>7.1f} m"
                      f"{sst:>11}"
                      f"{str(s['below_5m']) + '/' + str(s['impacts']):>11}")
            print(f"Virtual ground {FLOOR:g} m: going below it counts as FAILED (the real floor is never reached)")
        print(f"\nSaved: {base}.csv / _steps.csv / .json")
        link.land(timeout=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["flip", "spin", "bird"], choices=list(DISTURBANCES))
    ap.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-limits", action="store_true", help="do not touch the ATC_ACCEL limits")
    ap.add_argument("--compare-recovery", action="store_true",
                    help="fly every impact once per test (same hit): ardupilot / limit-on / limit-off / "
                         "spin-stop-fast / rate-rule")
    ap.add_argument("--modes", nargs="+", choices=RECOVERIES, default=None,
                    help="which tests to fly with --compare-recovery (default: all five)")
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    main(ap.parse_args())
