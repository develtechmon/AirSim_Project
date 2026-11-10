r"""
COLLECT RECOVERY DEMONSTRATIONS  (Stage 3 step 3a, ArduPilot SITL + AirSim)
==========================================================================

1. HOW TO RUN  (PowerShell in Stage_3, AirSim + SITL in GPS mode running, ~1.5-2 h)
------------------------------------------------------------------------
    Collect 400 hits:

    py .\collect_demonstration.py --hits 400

    If it stops (Ctrl+C) or SITL crashes - restart SITL, then continue where it stopped:

    py .\collect_demonstration.py --hits 400 --resume


2. EXPECTED RESULT
------------------------------------------------------------------------
    One line per hit, e.g.:
    Hit  12/400 | flip hard   | turns 2.6 | lowest  16.2 m | KEPT 74 samples | total 830 samples

    At the end:   ~95-100 % of hits KEPT,  ~20,000-30,000 samples in demos\recovery_demos.npz
    "lowest" should look like impact_check Test 6 (attitude-rule, start 30 m):
        Flip EASY / MEDIUM / HARD   22.7 / 19.1 / 16.6 m
        Bird EASY / MEDIUM / HARD   27.0 / 21.9 / 21.4 m
        Spin (all)                  ~30 m

    How to read it / what to do:
    Result                                  Meaning                      Action
    >= 90 % KEPT                            teacher works                send collect_episodes.csv -> next: train_imitation.py
    > 10 % "RESCUED - not kept"             teacher fails too often      stop, send collect_episodes.csv
    "REAL CRASH" printed                    SITL crashed                 restart SITL, run with --resume


3. OUTPUT FILES  (folder demos\, saved after EVERY hit - nothing is lost)
------------------------------------------------------------------------
    recovery_demos.npz       obs (N,13), act (N,3), episode (N,) - KEPT hits only, for BC
    collect_episodes.csv     one row per hit (type, level, intensity, lowest, kept, ...)
    collect_steps.csv        every 50 Hz step of every hit (also not kept), for plots
    collect_info.json        settings, rule parameters, observation / action definition


4. WHAT THE SCRIPT DOES  (same idea as Stage 1: expert flies -> we record -> BC copies)
------------------------------------------------------------------------
    Step 1  drone flies back to the start point (30 m) and holds 2 s
    Step 2  IMPACT: random type + level (same curriculum as training)
    Step 3  RECOVERY by the teacher = attitude-rule (your 3 phases):
            1 arrest the rotation  2 turn upright  3 brake the fall
            sent as ATTITUDE + THRUST -> ArduPilot's 400 Hz loop does the turning
            (the same command PPO will send, also on the real drone)
    Step 4  record what the drone SEES (observation) and what the teacher DOES (action)
    Step 5  KEEP the hit only if it recovered cleanly (no rescue, never below 5 m)


5. HIT MIX  (same curriculum as training)
------------------------------------------------------------------------
    Type    Share    Base turns     Level    Intensity    (turns = base turns x intensity)
    flip    45 %     2.0            EASY     0.7 - 0.9
    bird    45 %     1.0            MEDIUM   0.9 - 1.1
    spin    10 %     2.0            HARD     1.1 - 1.5
    Levels are picked 1/3 each.


6. WHAT IS RECORDED  (hardware-ready: only what the real drone measures)
------------------------------------------------------------------------
    OBSERVATION (13)                                   Sensor on the real drone
    0-2    "down" direction seen from the drone        IMU attitude
    3-5    body rates p, q, r  (rad/s)                 gyro
    6      falling speed (m/s, + = down)               EKF velocity
    7      height change since the hit (m)             EKF / baro
    8-9    sideways speed forward / right (m/s)        EKF velocity (GPS / flow)
    10-12  previous action                             network memory

    ACTION (3)
    0      target roll  (rad)
    1      target pitch (rad, heading frame)
    2      thrust / hover thrust   (Pi reads MOT_THST_HOVER -> same network fits any quad)


7. KEEP / NOT KEPT RULES
------------------------------------------------------------------------
    KEPT          recovered (rate < 1.2 rad/s, tilt < 30 deg, height > 3 m, speed < 3 m/s, held)
                  and never below the 5 m virtual ground
    RESCUED       about to go below 5 m -> script saves the drone (full thrust), NOT kept
    not kept      did not recover within the time limit
    REAL CRASH    script saves everything and stops -> restart SITL, --resume


8. SAFETY / SETTINGS
------------------------------------------------------------------------
    ATC_ACCEL_R/P/Y_MAX = 0 (no limit) and GUID_OPTIONS bit 3 ("thrust as thrust") ONLY from the
    hit until calm, then always put back to your values - also on Ctrl+C or a crash.
    4 crash-safety layers: rescue | save after every hit | stop cleanly on crash | --resume
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

import impact_check as ic
from ap_link import APLink, RateKeeper

OUT = Path("demos")
TYPES = {"flip": 0.45, "bird": 0.45, "spin": 0.10}
LEVELS = ["easy", "medium", "hard"]
OBS_N, ACT_N = 13, 3
EP_COLS = ["hit", "type", "level", "intensity", "turns_planned", "turns", "alt_end_impact_m", "lowest_m",
           "height_loss_m", "recovered", "recovery_s", "rescued", "kept", "samples", "drift_m"]
STEP_COLS = ["hit", "t_s", "phase", "alt_m", "tilt_deg", "rate_rads", "vdown_ms", "hspeed_ms",
             "roll_cmd", "pitch_cmd", "thr_x_hover"]


def observation(link, alt0, last_act):
    R = ic.quat_to_R(link.q)
    down = R[2, :]                                            # world down expressed in the body
    yaw = math.atan2(R[1, 0], R[0, 0])
    v = np.asarray(link.vel, dtype=float)
    fwd = math.cos(yaw) * v[0] + math.sin(yaw) * v[1]
    right = -math.sin(yaw) * v[0] + math.cos(yaw) * v[1]
    return np.concatenate([down, np.asarray(link.rates, float), [v[2], link.alt - alt0, fwd, right],
                           last_act]).astype(np.float32)


def teacher_action(link, hover):
    """attitude-rule -> (roll, pitch in the heading frame, thrust/hover) + the quaternion to send."""
    q, thr = ic.attitude_rule(link, hover)
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return np.array([roll, pitch, thr / hover], dtype=np.float32), q, thr


def one_hit(link, n, kind, level, rng, target, limits, guid, hover, steps_w):
    ic.fly_back(link, target, 2.0)
    p, q, r = ic.rates_for(kind, rng)
    intensity = rng.uniform(*ic.CURRICULUM[level])
    goal = ic.BASE_TURNS[kind] * intensity
    planned = goal * 2 * math.pi / float(np.linalg.norm([p, q, r]))
    alt0 = link.alt
    send = lambda vals: ic.send_params(link, vals)
    send({k: 0 for k in limits})                              # impact mode (hit) - stays off for recovery
    rk = RateKeeper(ic.HZ)
    t0 = t_prev = time.time()
    turned = 0.0
    while turned < goal and time.time() - t0 < ic.CAP * planned:   # ---- the hit
        link.send_rates(p, q, r, 0.5)
        rk.wait()
        link.update()
        now = time.time()
        turned += float(np.linalg.norm(link.rates)) * (now - t_prev) / (2 * math.pi)
        t_prev = now
        steps_w.writerow([n, round(now - t0, 3), "impact", round(link.alt, 2), round(link.tilt_deg, 1),
                          round(float(np.linalg.norm(link.rates)), 2), round(float(link.vel[2]), 2),
                          round(float(np.linalg.norm(link.vel[:2])), 2), "", "", ""])
        if ic.brake_floor(link) < ic.FLOOR + ic.MARGIN:
            break
        if ic.crashed(link):
            raise SystemExit("REAL CRASH during the hit - restart SITL, then run with --resume")
    alt_end = link.alt
    send({"GUID_OPTIONS": int(guid) | ic.GUID_THRUST_BIT})   # thrust = real thrust during recovery
    obs_l, act_l = [], []
    last = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    t_rel = time.time()
    lowest, ground, rescue, rule_on, rec_t, held = link.alt, False, False, True, None, None
    while time.time() - t_rel < ic.RECOVER_MAX_S:            # ---- the recovery
        o = observation(link, alt0, last)
        if rule_on:
            if rescue:
                link.send_rates(*ic.rate_rule(link, hover, full=True))
                a = None
            else:
                a, qt, thr = teacher_action(link, hover)
                ic.send_attitude(link, qt, thr)
                obs_l.append(o)
                act_l.append(a)
                last = a
        else:
            link.send_velocity(0.0, 0.0, 0.0)
        rk.wait()
        link.update()
        s = ic.state(link, target)
        lowest = min(lowest, s["alt"])
        doomed = s["alt"] - max(0.0, float(link.vel[2])) ** 2 / (2 * ic.RESCUE_BRAKE) < ic.RESCUE_FLOOR
        ground = ground or s["alt"] < ic.FLOOR or doomed
        if ground and not rescue:
            rescue = rule_on = True                           # FAILED - save the drone, not kept
        if ic.crashed(link):
            raise SystemExit("REAL CRASH during recovery - restart SITL, then run with --resume")
        steps_w.writerow([n, round(time.time() - t0, 3), "recovery", round(s["alt"], 2), round(s["tilt"], 1),
                          round(s["rate"], 2), round(float(link.vel[2]), 2), round(s["hspeed"], 2),
                          *(["", "", ""] if not (rule_on and obs_l and not rescue) else
                            [round(float(act_l[-1][0]), 3), round(float(act_l[-1][1]), 3),
                             round(float(act_l[-1][2]), 3)])])
        if rule_on and s["rate"] < ic.RATE_OK and s["tilt"] < ic.TILT_OK and float(link.vel[2]) < 1.0:
            rule_on = False                                   # calm: hand back to ArduPilot
            send({**dict(limits), "GUID_OPTIONS": int(guid)})
        if s["ok"]:
            held = held or time.time()
            if time.time() - held >= ic.HELD_S:
                rec_t = held - t_rel
                break
        else:
            held = None
    send({**dict(limits), "GUID_OPTIONS": int(guid)})         # always back to stock
    kept = bool(rec_t is not None and not ground and not rescue and obs_l)
    link.update()
    ep = [n, kind, level, round(intensity, 3), round(goal, 2), round(turned, 2), round(alt_end, 2),
          round(lowest, 2), round(alt0 - lowest, 2), rec_t is not None, round(rec_t, 2) if rec_t else "",
          rescue, kept, len(obs_l) if kept else 0, round(float(np.linalg.norm(link.pos - target)), 2)]
    return ep, (np.array(obs_l) if kept else None), (np.array(act_l) if kept else None)


def main(args):
    OUT.mkdir(exist_ok=True)
    npz, ep_csv, st_csv = OUT / "recovery_demos.npz", OUT / "collect_episodes.csv", OUT / "collect_steps.csv"
    obs_all, act_all, epi_all, done = np.zeros((0, OBS_N), np.float32), np.zeros((0, ACT_N), np.float32), \
        np.zeros(0, np.int32), 0
    if args.resume and ep_csv.exists():
        done = sum(1 for _ in open(ep_csv)) - 1
        if npz.exists():
            z = np.load(npz)
            obs_all, act_all, epi_all = z["obs"], z["act"], z["episode"]
        print(f"Resuming after {done} hits ({len(obs_all)} samples kept so far)")
    elif ep_csv.exists() and not args.resume:
        raise SystemExit("demos\\ already has data - use --resume to continue, or move the demos folder away")
    rng = np.random.RandomState(args.seed + done)
    link = APLink(args.conn, rate_hz=50)
    link.wait_state()
    limits = ic.read_limits(link)
    got = link.get_params(["GUID_OPTIONS", "MOT_THST_HOVER"])
    if not limits or "GUID_OPTIONS" not in got:
        raise SystemExit("Could not read ATC_ACCEL / GUID_OPTIONS from the FC - stopping")
    guid, hover = int(got["GUID_OPTIONS"]), float(got.get("MOT_THST_HOVER", 0.4))
    if any(abs(v) < 1e-6 for v in limits.values()) or guid & ic.GUID_THRUST_BIT:
        raise SystemExit(f"STOP - the FC still has test values from a crashed run: {limits} GUID_OPTIONS {guid}\n"
                         "Fix in MAVProxy: param set ATC_ACCEL_R_MAX 110000 / ATC_ACCEL_P_MAX 110000 / "
                         "ATC_ACCEL_Y_MAX 27000 / GUID_OPTIONS 0")
    info = {"teacher": "attitude-rule", "rule": {"version": ic.RULE_VERSION, "k": ic.RULE_K, "alpha": ic.RULE_ALPHA,
            "thr_inv_x_hover": ic.RULE_THR_INV, "up_min": ic.RULE_UP_MIN, "brake": ic.RULE_BRAKE,
            "side_k": ic.RULE_SIDE_K, "side_max": ic.RULE_SIDE_MAX}, "hover_thrust": hover, "hz": ic.HZ,
            "types": TYPES, "curriculum": ic.CURRICULUM, "base_turns": ic.BASE_TURNS, "start_alt_m": ic.ALT,
            "obs": ["down_x", "down_y", "down_z", "p", "q", "r", "v_down", "dalt_since_hit", "v_fwd", "v_right",
                    "prev_roll", "prev_pitch", "prev_thr_x_hover"],
            "act": ["roll_rad", "pitch_rad", "thrust_x_hover"], "limits_stock": limits, "guid_stock": guid,
            "yaw_brake_t": ic.YAW_BRAKE_T}
    json.dump(info, open(OUT / "collect_info.json", "w"), indent=2, default=str)
    print(f"Teacher: attitude-rule v{ic.RULE_VERSION} | hover thrust {hover:.2f} | target {args.hits} hits")
    new_ep, new_st = not ep_csv.exists(), not st_csv.exists()
    fe, fs = open(ep_csv, "a", newline=""), open(st_csv, "a", newline="")
    we, ws = csv.writer(fe), csv.writer(fs)
    if new_ep:
        we.writerow(EP_COLS)
    if new_st:
        ws.writerow(STEP_COLS)
    kinds, probs = list(TYPES), np.array(list(TYPES.values()))
    try:
        if not link.armed:
            link.takeoff_nogps(ic.ALT)
        link.update()
        target = link.pos.copy()
        target[2] = -ic.ALT
        for n in range(done + 1, args.hits + 1):
            kind = kinds[rng.choice(len(kinds), p=probs / probs.sum())]
            level = LEVELS[rng.randint(3)]
            ep, o, a = one_hit(link, n, kind, level, rng, target, limits, guid, hover, ws)
            we.writerow(ep)
            fe.flush()
            fs.flush()
            if o is not None:
                obs_all = np.concatenate([obs_all, o.astype(np.float32)])
                act_all = np.concatenate([act_all, a.astype(np.float32)])
                epi_all = np.concatenate([epi_all, np.full(len(o), n, np.int32)])
                np.savez(npz, obs=obs_all, act=act_all, episode=epi_all)
            print(f"Hit {n:3d}/{args.hits} | {kind:<4} {level:<6} | turns {ep[5]:.1f} | lowest {ep[7]:5.1f} m | "
                  + ("KEPT " + f"{ep[13]} samples" if ep[12] else ("RESCUED - not kept" if ep[11] else "not kept"))
                  + f" | total {len(obs_all)} samples")
    except KeyboardInterrupt:
        print("\nStopped by user - everything so far is saved. Continue with --resume")
    finally:
        ic.set_params(link, {**dict(limits), "GUID_OPTIONS": guid})
        fe.close()
        fs.close()
        print(f"\nSaved: {npz} ({len(obs_all)} samples) | {ep_csv} | {st_csv}")
        link.land(timeout=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", type=int, default=400)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yaw-brake", type=float, default=0.0,
                    help="yaw brake time constant (s), 0 = off. Default 0.0: the final model's teacher data was collected without it")
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    args = ap.parse_args()
    ic.YAW_BRAKE_T = args.yaw_brake
    main(args)
