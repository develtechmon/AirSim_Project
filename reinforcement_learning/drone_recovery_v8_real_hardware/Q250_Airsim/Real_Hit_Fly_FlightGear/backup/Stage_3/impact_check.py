r"""
IMPACT CHECK  (Stage 3, ArduPilot SITL + AirSim)
================================================

1. HOW TO RUN  (PowerShell in Stage_3, AirSim + SITL in GPS mode running)
------------------------------------------------------------------------
    Main test - compare the recoveries (~15 min):

    py .\impact_check.py --repeats 2 --compare-recovery --modes spin-stop-fast attitude-rule rate-rule

    Quick test - one hard flip, ArduPilot only (~2 min):

    py .\impact_check.py --types flip --levels hard

    All 3 types x 3 levels, ArduPilot only (~5 min):

    py .\impact_check.py


2. EXPECTED RESULT  (summary table, "Avg low" = average lowest height after the hit,
   start 30 m, higher = better; measured 2026-10-19, Test 6, about +-1 m per hit)
------------------------------------------------------------------------
    Recovery         Flip EASY  MEDIUM  HARD  | Bird EASY  MEDIUM  HARD  | Spin  | Below 5 m
    attitude-rule      22.7     19.1    16.6  |   27.0     21.9    21.4  |  ~30  |    0
    spin-stop-fast     22.9     18.4    17.0  |   26.7     24.7    19.8  |  ~30  |    0
    rate-rule (v3)     19.9     16.2    12.9  |   25.0     23.2    18.8  |  ~30  |    0
    ardupilot (stock)  HARD flip 3.0 turns: FAILED 6 of 8, lowest 0.4 - 5.1 m


    How to read it: attitude-rule ~= spin-stop-fast  -> attitude-rule is the BC teacher.
                    rate-rule is lower everywhere   -> 50 Hz delay overshoot (55-97 deg).


3. OUTPUT FILES  (folder runs\impact_check\)
------------------------------------------------------------------------
    impact_check_<date>.csv         one line per impact
    impact_check_<date>_steps.csv   every 50 Hz step (for plots)
    impact_check_<date>.json        settings, limits used, summary


4. HOW THE IMPACT IS MADE  (same in SITL and on the real drone)
------------------------------------------------------------------------
    Step 1  drone hovers at 30 m in GUIDED
    Step 2  send BODY-RATE commands ("tumble this fast") until the gyro has
            counted the planned number of turns           = the IMPACT
    Step 3  stop the tumble, the chosen RECOVERY takes over


5. IMPACT LEVELS  (same as the Stage 3 training curriculum)
------------------------------------------------------------------------
    turns = base turns x intensity      intensity: EASY 0.7-0.9 | MEDIUM 0.9-1.1 | HARD 1.1-1.5

    Type   Axis                    Rate (rad/s)   Base   EASY      MEDIUM    HARD
    flip   roll OR pitch (random)  16             2.0    1.4-1.8   1.8-2.2   2.2-3.0 turns
    spin   yaw                     10             2.0    1.4-1.8   1.8-2.2   2.2-3.0 turns
    bird   roll + pitch + yaw      10 / 10 / 6    1.0    0.7-0.9   0.9-1.1   1.1-1.5 turns

    1 turn = 360 deg. Direction random from --seed (a rerun is identical).
    Safety cap: an impact never lasts longer than 3x its planned time.


6. THE RECOVERIES  (--modes; every mode flies the SAME hit, only the recovery changes)
------------------------------------------------------------------------
    Name            Test  What recovers the drone                                   Role
    ardupilot       1     ArduPilot alone, "hold still"                             stock baseline
    limit-on        2     velocity "brake + climb 5 m/s", WPNAV limits stock        old design (did not help)
    limit-off       3     same, WPNAV limits raised                                 old design (did not help)
    spin-stop-fast  4     ArduPilot, rotation limits (ATC_ACCEL) OFF until          BAR PPO MUST BEAT
                          spin < 3 rad/s
    rate-rule       5     hand-made rule, BODY-RATE + thrust at 50 Hz               overshoots (delay)
    attitude-rule   6     same rule, sent as ATTITUDE + thrust, ArduPilot's          BC TEACHER
                          400 Hz loop does the turning                              (= what PPO will send)
    bc-policy       7     the trained BC network - run it with bc_impact_check.py   BC CHECK

    rate-rule / attitude-rule (your 3 phases):
        1 arrest the rotation   2 turn upright (hover thrust while upside down)   3 brake the fall, lean against drift
    During the recovery ONLY: ATC_ACCEL off + GUID_OPTIONS bit 3 ("thrust as thrust"),
    then hand back to ArduPilot when calm (rate < 1.2 rad/s, tilt < 30 deg, falling < 1 m/s).


7. PASS / FAIL RULES
------------------------------------------------------------------------
    RECOVERED       all at once for 0.5 s: rate < 1.2 rad/s, tilt < 30 deg,
                    height > 3 m, horizontal speed < 3 m/s
    FAILED          went below the VIRTUAL GROUND (5 m), or could no longer stop above it
    RESCUE          after a FAIL the script saves the drone itself (full-thrust rule),
                    so SITL never really crashes - the hit still counts as FAILED
    REAL CRASH      disarmed in the air / on the floor: script saves, stops and says so
    Impact cut      tumble stopped early if the drone could no longer brake above 5 m + 2 m


8. SAFETY / SETTINGS
------------------------------------------------------------------------
    ATC_ACCEL_R/P/Y_MAX set to 0 (no limit) only during the hit (and the recovery for modes
    4-6), always put back to your values after - also on Ctrl+C. --keep-limits = never touch them.
    GUID_OPTIONS always put back to your value (0).
    Before every hit the drone flies back to the start point and holds 3 s.
    "[fc] Potential Thrust Loss" / "Vibration compensation" / "GPS Glitch" during a flip are
    normal: a motor is at full power while tumbling.
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
BASE_TURNS = {"flip": 2.0, "spin": 2.0, "bird": 1.0}       # same as disturbance_injector.py
CURRICULUM = {"easy": (0.7, 0.9), "medium": (0.9, 1.1), "hard": (1.1, 1.5)}   # same as the training
TURNS = {k: {lv: f"{BASE_TURNS[k] * lo:g}-{BASE_TURNS[k] * hi:g}" for lv, (lo, hi) in CURRICULUM.items()}
         for k in BASE_TURNS}                           # for printing / the json
LEVELS = ["easy", "medium", "hard"]
RECOVERIES = ["ardupilot", "limit-on", "limit-off", "spin-stop-fast", "rate-rule", "attitude-rule", "bc-policy"]
RULES = ("rate-rule", "attitude-rule", "bc-policy")
BC_MAX_LEAN = 0.6          # rad, clip on the network's roll / pitch target (teacher uses <= 0.45)
_BC = {}
SPIN_STOPPED = 3.0         # rad/s: spin-stop-fast keeps the rotation limits off until below this
RULE_VERSION = 3           # rate-rule v3. v1: 15% thrust while tilted -> 2-4.5 m below spin-stop-fast.
                           # v2: 0.4 x hover while tilted -> 1-3 m below: too little thrust to STOP the rotation,
                           #     so it overshot upright (25 -> 70 deg back) and lost height while re-levelling.
RULE_K = 6.0               # rad/s of turn rate per rad of attitude error
RULE_MAX = 15.0            # rad/s max turn rate it asks for
RULE_THR_INV = 1.0         # x hover: thrust while tilted past ~73 deg / upside down (motors need it to brake
                           #          the rotation; v2 used 0.4 and overshot)
RULE_ALPHA = 30.0          # rad/s^2: turn rate is limited to sqrt(2 x ALPHA x error) so it can stop AT upright
RULE_UP_MIN = 0.3          # "upright enough to push": cos(tilt) above this (tilt < ~73 deg)
RULE_BRAKE = 0.25          # x hover: extra vertical thrust per m/s of falling speed
RULE_SIDE_K = 0.15         # rad of lean per m/s of sideways speed, to brake the drift ...
RULE_SIDE_MAX = 0.45       # ... at most tan(~24 deg); only once nearly upright (cos(tilt) > 0.7)
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


def quat_to_R(q):
    """Rotation matrix body -> world (NED) from ArduPilot's quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rate_rule(link, hover, full=False):
    """rate-rule v2 - the 3 phases with rotation-rate + thrust commands (thrust relative to hover):
    1 arrest + turn upright: rotate the shortest way to the target attitude (6 rad/s per rad, max 15)
    2 keep steering: while tilted past ~73 deg / upside down, thrust = hover (enough for the motors to
      brake the rotation), turn rate limited so it stops AT upright (no overshoot)
    3 brake: once upright enough, vertical thrust = hover x (1 + 0.25 x falling speed), divided by
      cos(tilt) so the UPWARD part is what we want (max 100%); when nearly upright the target
      attitude leans against the sideways speed to stop the drift."""
    R = quat_to_R(link.q)
    up = float(R[2, 2])                                      # cos(tilt): 1 upright, -1 upside down
    vel = np.asarray(link.vel, dtype=float)
    target_down = np.array([0.0, 0.0, 1.0])                  # world: body-down axis should point here
    if up > 0.7:                                             # nearly upright: lean against the drift
        lean = np.clip(RULE_SIDE_K * vel[:2], -RULE_SIDE_MAX, RULE_SIDE_MAX)
        target_down = np.array([lean[0], lean[1], 1.0])
        target_down /= np.linalg.norm(target_down)
    d = R.T @ target_down                                    # the target "down" seen from the drone
    axis = np.array([-d[1], d[0], 0.0])                      # body-down x target-down
    sn = float(np.linalg.norm(axis))
    ang = math.atan2(sn, d[2])                               # attitude error 0..pi
    u = axis / sn if sn > 1e-3 else (np.array([1.0, 0.0, 0.0]) if d[2] < 0 else np.zeros(3))
    wr = min(RULE_K * ang, RULE_MAX, math.sqrt(2 * RULE_ALPHA * ang)) * u   # no overshoot past upright
    v_down = max(0.0, float(vel[2]))
    if up < RULE_UP_MIN:
        thr = hover if full else RULE_THR_INV * hover        # tilted / upside down: keep steering only
    elif full:
        thr = 1.0                                            # rescue: full brake as soon as upright enough
    else:
        thr = float(np.clip(hover * (1 + RULE_BRAKE * v_down) / up, RULE_THR_INV * hover, 1.0))
    return float(wr[0]), float(wr[1]), 0.0, thr


def rule_targets(link, hover, full=False):
    """Shared by both rules: target attitude (level, or leaning against the sideways drift once nearly
    upright) and the v3 thrust schedule (relative to hover). Returns (R, target_down_world, thrust)."""
    R = quat_to_R(link.q)
    up = float(R[2, 2])
    vel = np.asarray(link.vel, dtype=float)
    target_down = np.array([0.0, 0.0, 1.0])
    if up > 0.7:
        lean = np.clip(RULE_SIDE_K * vel[:2], -RULE_SIDE_MAX, RULE_SIDE_MAX)
        target_down = np.array([lean[0], lean[1], 1.0]) / math.sqrt(1 + lean[0] ** 2 + lean[1] ** 2)
    v_down = max(0.0, float(vel[2]))
    if up < RULE_UP_MIN:
        thr = hover if full else RULE_THR_INV * hover
    elif full:
        thr = 1.0
    else:
        thr = float(np.clip(hover * (1 + RULE_BRAKE * v_down) / up, RULE_THR_INV * hover, 1.0))
    return R, target_down, thr


def attitude_rule(link, hover, full=False):
    """attitude-rule: the SAME targets and thrust as rate-rule v3, but sent as an ATTITUDE target
    (level / lean angle + thrust). ArduPilot's own 400 Hz attitude loop then turns the drone - no
    50 Hz MAVLink delay inside the fast rotation-stopping loop. Returns (quaternion w,x,y,z, thrust)."""
    R, td, thr = rule_targets(link, hover, full)
    yaw = math.atan2(R[1, 0], R[0, 0])                       # keep the current heading
    cy, sy = math.cos(yaw), math.sin(yaw)
    lx, ly = td[0] / td[2], td[1] / td[2]                    # lean (world north/east) ...
    fx, fy = cy * lx + sy * ly, -sy * lx + cy * ly           # ... in the heading frame
    pitch, roll = math.atan(fx), -math.atan(fy)              # body-down toward +forward = nose up
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cyh, syh = math.cos(yaw / 2), math.sin(yaw / 2)
    q = (cr * cp * cyh + sr * sp * syh, sr * cp * cyh - cr * sp * syh,
         cr * sp * cyh + sr * cp * syh, cr * cp * syh - sr * sp * cyh)
    return q, thr


def send_attitude(link, q, thrust):
    """SET_ATTITUDE_TARGET with the attitude quaternion + thrust (body rates ignored: type_mask 7)."""
    with link.io_lock:
        link.m.mav.set_attitude_target_send(0, link.m.target_system, link.m.target_component, 7,
                                            [float(v) for v in q], 0.0, 0.0, 0.0, float(thrust))


def bc_observation(link, alt0, last):
    """The SAME 13 inputs as collect_demonstration.py (down 3, gyro 3, v_down, dalt, v_fwd, v_right, last action)."""
    R = quat_to_R(link.q)
    yaw = math.atan2(R[1, 0], R[0, 0])
    v = np.asarray(link.vel, dtype=float)
    fwd = math.cos(yaw) * v[0] + math.sin(yaw) * v[1]
    right = -math.sin(yaw) * v[0] + math.cos(yaw) * v[1]
    return np.concatenate([R[2, :], np.asarray(link.rates, float), [v[2], link.alt - alt0, fwd, right],
                           last]).astype(np.float32)


def load_bc():
    """Load the BC network BEFORE flying: importing torch takes several seconds - doing that in the
    middle of a recovery leaves the drone without commands (first test: it fell 30 m upside down)."""
    import torch
    from train_imitation import RecoveryPolicy, normalize
    t0 = time.time()
    net = RecoveryPolicy()
    net.load_state_dict(torch.load("models/recovery_policy_best.pth", map_location="cpu"))
    net.eval()
    z = np.load("models/obs_norm.npz")
    _BC.update(net=net, torch=torch, norm=normalize, mean=z["mean"], std=z["std"])
    with torch.no_grad():                                    # warm-up: first call is slow too
        net(torch.zeros(1, len(z["mean"])))
    print(f"BC network loaded in {time.time() - t0:.1f} s (models\\recovery_policy_best.pth)")


def bc_policy(link, hover, alt0, last):
    """bc-policy: the trained BC network (models\\recovery_policy_best.pth) gives roll, pitch, thrust x hover;
    sent as an ATTITUDE target + thrust exactly like attitude-rule. Returns (quaternion, thrust, action)."""
    if not _BC:
        load_bc()
    x = _BC["norm"](bc_observation(link, alt0, last), _BC["mean"], _BC["std"])
    with _BC["torch"].no_grad():
        a = _BC["net"](_BC["torch"].as_tensor(x[None]))[0].numpy().astype(np.float32)
    roll = float(np.clip(a[0], -BC_MAX_LEAN, BC_MAX_LEAN))
    pitch = float(np.clip(a[1], -BC_MAX_LEAN, BC_MAX_LEAN))
    thr = float(np.clip(a[2] * hover, 0.0, 1.0))
    R = quat_to_R(link.q)
    yaw = math.atan2(R[1, 0], R[0, 0])                       # keep the current heading
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cyh, syh = math.cos(yaw / 2), math.sin(yaw / 2)
    q = (cr * cp * cyh + sr * sp * syh, sr * cp * cyh - cr * sp * syh,
         cr * sp * cyh + sr * cp * syh, cr * cp * syh - sr * sp * cyh)
    return q, thr, a


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
               guid=None, hover=0.4, goal=None):
    fly_back(link, target, HOLD_S)
    p, q, r = rates
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
    keep_rot = recovery in ("spin-stop-fast",) + RULES and bool(limits)
    if limits and not keep_rot:
        send_params(link, limits)                        # stock ArduPilot for the recovery
    rule_on = recovery in RULES
    if rule_on:
        send_params(link, {"GUID_OPTIONS": int(guid) | GUID_THRUST_BIT})   # thrust field = real thrust
    spin_stop_t = rule_end_t = None
    rescue = False
    t_rel = time.time()
    # ---- recovery: ardupilot = hold still (baseline); limit-on / limit-off = brake + climb hard
    rec_t, held_since, min_alt, ground = None, None, link.alt, False
    alt_end_impact = link.alt
    bc_last = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    while time.time() - t_rel < RECOVER_MAX_S:
        if rule_on:
            if recovery == "attitude-rule" and not rescue:
                send_attitude(link, *attitude_rule(link, hover))
            elif recovery == "bc-policy" and not rescue:
                q_bc, thr_bc, bc_last = bc_policy(link, hover, alt0, bc_last)
                send_attitude(link, q_bc, thr_bc)
            else:
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
                         ((recovery in RULES or rescue) and not rule_on)):
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
            "planned_turns": round(goal, 2), "turns": round(turned, 2), "impact_s": round(impact_s, 2),
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
              ", ".join(f"{lv} {TURNS[k][lv]}" for lv in args.levels) + " turns")
    modes = (args.modes or RECOVERIES) if args.compare_recovery else ["ardupilot"]
    if args.compare_recovery:
        print(f"Compare recovery: every impact flown {len(modes)}x - {', '.join(modes)} (same hit each time)")
    print(f"Recovery (baseline): ArduPilot GUIDED hold | criterion: rate < {RATE_OK} rad/s, "
          f"tilt < {TILT_OK:.0f} deg, alt > {ALT_OK:.0f} m, h-speed < {HSPEED_OK:.0f} m/s for {HELD_S} s")
    print()
    link = APLink(args.conn, rate_hz=50)
    link.wait_state()
    original = read_limits(link)
    stuck = {k: v for k, v in original.items() if abs(v) < 1e-6}
    g0 = link.get_params(["GUID_OPTIONS"]).get("GUID_OPTIONS")
    if stuck or (g0 is not None and int(g0) & GUID_THRUST_BIT):
        raise SystemExit("STOP - the FC still has test values from a crashed run: "
                         f"{stuck or ''} {'GUID_OPTIONS ' + str(int(g0)) if g0 is not None and int(g0) & GUID_THRUST_BIT else ''}\n"
                         "Fix in the MAVProxy / SITL console, then run again:\n"
                         "    param set ATC_ACCEL_R_MAX 110000\n    param set ATC_ACCEL_P_MAX 110000\n"
                         "    param set ATC_ACCEL_Y_MAX 27000\n    param set GUID_OPTIONS 0")
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
    if any(m in RULES for m in modes) and (guid is None or not limits):
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
    base = OUT / f"{OUT.name}_{stamp}"
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
                    goal = BASE_TURNS[kind] * rng.uniform(*CURRICULUM[level])
                    for mode in modes:
                        n += 1
                        e = one_impact(link, kind, level, rates, n, rows, target, limits, mode, free, guid, hover,
                                       goal)
                        results.append(e)
                        spin = (f" | spin stopped {e['spin_stopped_s']:.2f}s after hit"
                                if e["spin_stopped_s"] is not None else " | spin not stopped")
                        rec = (f"recovered {e['recovery_time_s']:.2f}s" if e["recovered"] else
                               "FAILED - hit virtual ground" if e["virtual_ground"] else "FAILED - not recovered")
                        tag = f" | {mode:<14}" if args.compare_recovery else ""
                        print(f"Impact {n:2d} | {kind:<4} {level:<6}{tag} | turns {e['turns']:.1f}/{e['planned_turns']:.1f} "
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
                   "limit_off_values": {k: FREE_VALUES[k] for k in free}, "climb_mps": CLIMB, "rate_rule": {"version": RULE_VERSION, "k": RULE_K, "max_rate": RULE_MAX,
                   "thr_inv_x_hover": RULE_THR_INV, "alpha": RULE_ALPHA, "up_min": RULE_UP_MIN, "brake": RULE_BRAKE,
                   "side_k": RULE_SIDE_K, "side_max": RULE_SIDE_MAX}, "impact_mode": bool(limits), "hover_alt_m": ALT,
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
                    help="which tests to fly with --compare-recovery (default: all seven)")
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    main(ap.parse_args())
