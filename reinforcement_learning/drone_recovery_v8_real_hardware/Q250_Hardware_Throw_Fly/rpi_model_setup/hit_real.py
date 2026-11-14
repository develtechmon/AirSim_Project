r"""
hit_real.py - IN-FLIGHT HIT TEST: you hover the 5-inch Q250, flip a switch, the Pi hits it and the PPO recovers it
=====================================================================================================
The real-drone twin of impact_check.py (SITL, 12 Nov 2026: PPO + Option B recovered 12/12 hits at 30 m).
Same Pi, wiring, model and link as throw_real.py (it imports its helpers - keep both files in the folder).
Version 12 Nov 2026 (a): first version. Recovery = PPO + Option B (PPO levels, height hold brakes, ArduPilot holds).

0. SETTINGS AT THE TOP (change here only)
------------------------------------------------------------------------------------------------------
    | Setting      | Value                | Meaning                                                      |
    |--------------|----------------------|--------------------------------------------------------------|
    | HIT_CH       | 11 flip, 12 bird,    | switch -> hit type (2-position switches; ON above 1700)      |
    |              | 13 spin              |                                                              |
    | MIN_HEIGHT_M | 20                   | m above home: refuse a hit below this (sim worst loss 11 m)  |
    | CUT_HEIGHT_M | 12                   | m: stop the hit early if it could not brake above this       |
    | RESCUE_M     | 6                    | m: below this (braking included) the rule rescues at full    |
    |              |                      | thrust - the hit then counts as FAILED                       |
    | HOLD_S       | 10                   | s GUIDED hold after the recovery, then back to YOUR mode     |
    | MIN_VOLT     | 14.4                 | V in the air under load (4S, 3.6 V per cell)                 |

1. HOW TO RUN  (Pi, folder Q250_Hardware_Throw_Fly, ALWAYS inside tmux; stop the throw autostart first)
------------------------------------------------------------------------------------------------------
    sudo systemctl stop throw

    tmux new -s hit

    cd ~/Q250_Hardware_Throw_Fly

    python3 hit_real.py --check                    (bench: link, the 3 switches, buzzer - never arms, never moves)

    python3 hit_real.py                            (easy hits, PPO + Option B - start here)

    python3 hit_real.py --level medium             (later: medium, then hard)

    python3 hit_real.py --mode attitude-rule       (the rule, for comparison - same hits)

    SITL test (WSL, simulator running): add --conn tcp:127.0.0.1:5762 --baud 115200 - the ONLY difference.
    After the session: sudo systemctl start throw   (throw autostart back on)

2. THE BUZZER  (same tunes as throw_real.py)
------------------------------------------------------------------------------------------------------
    | You hear          | Means                                            | Do                            |
    |-------------------|--------------------------------------------------|-------------------------------|
    | 1 long + 2 short  | Pi started, PPO (1 long + 1 short = rule)        | take off yourself, PosHold    |
    | 3 short beeps     | checks passed - the hit starts NOW               | hands off the sticks (middle) |
    | 2 long beeps      | REFUSED (see printout) or hit ended / you took   | read the console later        |
    |                   | over                                             |                               |

3. ONE HIT  (you fly; the Pi only acts between your switch and the hand-back)
------------------------------------------------------------------------------------------------------
    | # | Who      | Does                                                      | Script                       |
    |---|----------|-----------------------------------------------------------|------------------------------|
    | 1 | You      | take off, climb to >= 25 m, CH8 POSHOLD, sticks middle    | WAIT (prints height)         |
    | 2 | You      | flip ONE hit switch ON (CH11 flip / CH12 bird / CH13 spin)| 6 checks -> 3 short beeps    |
    | 3 | Pi       | GUIDED; hit = spin the drone N turns (rotation limits off)| "HIT flip easy 1.6 turns"    |
    | 4 | Pi       | recovery: PPO levels -> height hold brakes the fall        | 50 Hz, Option B              |
    |   |          |  (attitude-rule: the rule does both)                       |                              |
    | 5 | ArduPilot| GUIDED position hold 10 s                                  | "[hold] ..."                 |
    | 6 | Pi       | back to YOUR mode (POSHOLD), 2 long beeps                  | "back to you in POSHOLD"     |
    | 7 | You      | switch OFF; next hit: switch ON again                      | result line, files saved     |

4. CHECKS WHEN A HIT SWITCH GOES ON  (all must pass, else "REFUSED: ..." + 2 long beeps)
------------------------------------------------------------------------------------------------------
    | # | Check                  | Needed                        | Why                                      |
    |---|------------------------|-------------------------------|------------------------------------------|
    | 1 | armed, flying          | landed_state "in air"         | never hits a drone on the ground         |
    | 2 | your mode              | POSHOLD or LOITER             | the Pi hands back to this mode           |
    | 3 | height above home      | >= MIN_HEIGHT_M (20 m)        | sim: worst loss 11 m with Option B       |
    | 4 | sticks                 | throttle 1400-1600 (middle)   | PosHold holds height when handed back    |
    | 5 | GPS / EKF              | 3D fix >= 5 sats, EKF GPS     | GUIDED hold needs a position             |
    | 6 | battery                | >= 14.4 V (under load)        | enough thrust to brake                   |

5. TAKE OVER / ABORT  (any time)
------------------------------------------------------------------------------------------------------
    | You do / happens                | Result                                                      |
    |---------------------------------|-------------------------------------------------------------|
    | CH8 to ANOTHER position, e.g.   | the script stops at once, puts the parameters back - YOU fly |
    |  AltHold, then back to PosHold  |  (CH8 is still on PosHold, so it must MOVE to be seen)       |
    | CH7 (disarm)                    | motors off (ArduPilot) - only if it is about to hit something |
    | hit switch OFF during the hit   | NOT an abort (the hit is 0.5-4 s); use CH8                   |
    | could not brake above 12 m      | hit stopped early ("cut"), recovery starts at once          |
    | would go below 6 m              | RESCUE: rule at full thrust; the hit counts as FAILED       |
    | not calm 8 s after the hit      | GUIDED hold anyway (ArduPilot levels it)                     |

6. OUTPUTS  (runs/hit_real/, written + flushed after EVERY hit)
------------------------------------------------------------------------------------------------------
    hit_real_<date>.csv        one line per hit: type, level, turns, impact_s, peak rate/tilt, height before,
                               lowest, height lost, b_handover_s, recovered_s, rescued, ended
    hit_real_<date>_steps.csv  every 50 Hz step: phase (hit / ppo / hold-brake / rule / rescue / hold), tilt,
                               rate, height above home, speeds, p q r
    hit_real_<date>.json       settings + every hit.          + the FC's own .bin log (Mission Planner)

7. EXPECTED  (SITL 12 Nov 2026: SITL 1e001fa0, q250.json, sensor 2 cm off-centre, THIS script with your
   switches - CH9 arm, CH8 PosHold, CH11/12/13 - hover 30 m; guide "Q250 Recovery Option B")
------------------------------------------------------------------------------------------------------
    | Hit           | PPO + Option B: height lost        | attitude-rule | After the hit               |
    |---------------|------------------------------------|---------------|-----------------------------|
    | flip          | easy 4.2 / medium 6.3 / hard 6.4 m | hard 4.5 m    | GUIDED hold 10 s, then back |
    | bird          | easy 3.8 / medium 11.5 / hard 5.1 m| hard 0.6 m    | to POSHOLD every time       |
    | spin          | easy 3.6 / medium 3.8 / hard 4.3 m | hard 0.4 m    |                             |
    All 12 recovered (9 PPO + 3 rule), no rescue. Also tested: CH11 at 15 m -> "REFUSED: height ...";
    take-over (CH8 AltHold -> PosHold 1 s into the recovery) -> "you took over (ALT_HOLD) - the script lets go".
    Same hits in impact_check.py (12 per method): PPO alone 6/12 (avg 12.2 m lost), PPO + Option B 12/12
    (avg 5.3 m, worst 11.0 m), attitude-rule 12/12 (avg 2.5 m).
"""

import argparse
import csv
import datetime
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np

import impact_check as ic
import throw_real as tr
from ap_link import RateKeeper, M

HIT_CH = {11: "flip", 12: "bird", 13: "spin"}
MIN_HEIGHT_M = 20.0        # m above home: refuse below (SITL worst loss with Option B: 11 m)
CUT_HEIGHT_M = 12.0        # m: stop the hit early if braking could not stop above this
RESCUE_M = 6.0             # m: rescue (full-thrust rule) if it would pass this even braking
HOLD_S = 10.0              # s GUIDED hold after the recovery, then back to the pilot's mode
RECOVER_MAX_S = 8.0        # s: not calm by then -> GUIDED hold anyway
MIN_VOLT = 14.4            # V in the air under load (4S)
PILOT_MODES = ("POSHOLD", "LOITER")
STICK_MID = (1400, 1600)   # throttle stick in the middle (PosHold holds height)
HZ = 50.0
OUT = Path("runs") / "hit_real"


def rel_height(link):
    """Height above home (m) from the FC (GLOBAL_POSITION_INT.relative_alt), None if not received."""
    m = link.msgs.get("GLOBAL_POSITION_INT")
    return None if m is None else m.relative_alt / 1000.0


def checks(link):
    fix, sats = tr.gps(link)
    h = rel_height(link)
    out = [("armed, flying", link.armed and tr.landed_state(link) == 2,
            f"{'armed' if link.armed else 'DISARMED'}, {'in air' if tr.landed_state(link) == 2 else 'ON GROUND'}"),
           ("your mode POSHOLD / LOITER", link.mode in PILOT_MODES, link.mode),
           (f"height >= {MIN_HEIGHT_M:.0f} m above home", h is not None and h >= MIN_HEIGHT_M, f"{h:.1f} m" if h is not None else "no data"),
           ("throttle stick middle", STICK_MID[0] <= tr.rc(link, 3) <= STICK_MID[1], f"CH3 {tr.rc(link, 3)}"),
           ("GPS 3D fix + EKF GPS", fix >= 3 and sats >= tr.MIN_SATS and tr.ekf_gps(link), f"fix {fix}, {sats} sats, EKF {'yes' if tr.ekf_gps(link) else 'NO'}"),
           (f"battery >= {MIN_VOLT} V", tr.volts(link) >= MIN_VOLT, f"{tr.volts(link):.2f} V")]
    return all(c[1] for c in out), out


def log_row(rows, n, kind, phase, t, link, h0, alt0):
    rows.append([n, kind, phase, round(t, 3), round(link.tilt_deg, 1), round(float(np.linalg.norm(link.rates)), 2),
                 round(h0 + link.alt - alt0, 2), round(float(np.linalg.norm(link.vel[0:2])), 2), round(float(link.vel[2]), 2)]
                + [round(float(v), 2) for v in link.rates])


def one_hit(link, kind, level, mode, n, rows, hover, guid, limits, ff, net, rng):
    """Hit (lead hit, rotation limits off) -> recovery (PPO + Option B, or attitude-rule) -> GUIDED hold -> back."""
    pilot_mode, h0, alt0 = link.mode, rel_height(link), link.alt
    hgt = lambda: h0 + link.alt - alt0                           # height above home, 50 Hz (EKF)
    p, q, r = ic.rates_for(kind, rng)
    goal = ic.BASE_TURNS[kind] * rng.uniform(*ic.CURRICULUM[level])
    planned = goal * 2 * math.pi / float(np.linalg.norm([p, q, r]))
    tr.beep(link, tr.TUNE_READY)
    if not tr.request_mode(link, "GUIDED", timeout=2):
        tr.beep(link, tr.TUNE_REFUSED)
        print(f"REFUSED by ArduPilot: GUIDED not accepted (mode {link.mode})")
        return None
    print(f"HIT {kind} {level} {goal:.1f} turns | rates {p:+.0f} {q:+.0f} {r:+.0f} rad/s | from {h0:.1f} m | {mode}")
    rk = RateKeeper(HZ)
    t0 = t_prev = time.time()
    turned = peak_rate = peak_tilt = 0.0
    low, cut, why, took_over = h0, False, None, False
    if limits:
        ic.send_params(link, {k: 0 for k in limits})              # rotation limits off for the hit
    if ff is not None:
        ic.send_params(link, {ic.FF_NAME: 0})
    try:
        # ---- the hit
        while turned < goal and time.time() - t0 < ic.CAP * planned:
            ic.send_hit(link, (p, q, r))
            rk.wait()
            link.update()
            now = time.time()
            turned += float(np.linalg.norm(link.rates)) * (now - t_prev) / (2 * math.pi)
            t_prev = now
            peak_rate, peak_tilt, low = max(peak_rate, float(np.linalg.norm(link.rates))), max(peak_tilt, link.tilt_deg), min(low, hgt())
            log_row(rows, n, kind, "hit", now - t0, link, h0, alt0)
            if link.mode != "GUIDED" or not link.armed:
                took_over = True
                break
            if hgt() - max(0.0, float(link.vel[2])) ** 2 / (2 * ic.BRAKE) < CUT_HEIGHT_M:
                cut = True
                break
        impact_s = time.time() - t0
        if ff is not None:
            ic.send_params(link, {ic.FF_NAME: ff})                # your value back BEFORE the recovery
        # ---- the recovery
        t_rec, b_t, rec_s, rescue, held, phase = time.time(), None, None, False, None, "ppo" if mode == "ppo-policy" else "rule"
        last = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if not took_over:
            ic.send_params(link, {"GUID_OPTIONS": int(guid) | ic.GUID_THRUST_BIT})   # thrust = real thrust
        while not took_over and time.time() - t_rec < RECOVER_MAX_S:
            rate, tilt, h, vz = float(np.linalg.norm(link.rates)), link.tilt_deg, hgt(), float(link.vel[2])
            if not rescue and h - max(0.0, vz) ** 2 / (2 * ic.RESCUE_BRAKE) < RESCUE_M:
                rescue, phase = True, "rescue"
                print(f"[rescue] would pass {RESCUE_M:.0f} m - full-thrust rule (counts as FAILED)")
            if phase == "ppo" and rate < ic.RATE_OK and tilt < ic.TILT_OK:
                phase, b_t = "hold-brake", time.time() - t_rec       # Option B: level -> height hold brakes
            if rescue:
                link.send_rates(*ic.rate_rule(link, hover, full=True))
            elif phase == "ppo":
                qa, thr, last = tr.ppo_step(net, link, hover, alt0, last)
                ic.send_attitude(link, qa, thr)
            elif phase == "rule":
                ic.send_attitude(link, *ic.attitude_rule(link, hover))
            else:
                ic.send_attitude(link, *tr.height_hold(link, hover))
            rk.wait()
            link.update()
            low = min(low, hgt())
            peak_rate = max(peak_rate, float(np.linalg.norm(link.rates)))
            log_row(rows, n, kind, phase, time.time() - t0, link, h0, alt0)
            if link.mode != "GUIDED" or not link.armed:
                took_over = True
                break
            calm = (float(np.linalg.norm(link.rates)) < ic.RATE_OK and link.tilt_deg < ic.TILT_OK
                    and abs(float(link.vel[2])) < tr.VZ_OK)
            held = (held or time.time()) if calm else None
            if held and time.time() - held >= ic.HELD_S:
                rec_s = held - t_rec
                break
        why = "you took over" if took_over else ("recovered" if rec_s is not None else f"not calm in {RECOVER_MAX_S:.0f} s")
    finally:
        ic.send_params(link, {"GUID_OPTIONS": int(guid)})
        if limits:
            ic.send_params(link, limits)
        if ff is not None:
            ic.send_params(link, {ic.FF_NAME: ff})
    # ---- GUIDED hold, then back to the pilot's mode
    held_s = 0.0
    if not took_over and link.armed and link.mode == "GUIDED":
        print(f"[hold] GUIDED position hold {HOLD_S:.0f} s ({why})")
        t_h = time.time()
        while time.time() - t_h < HOLD_S and link.mode == "GUIDED" and link.armed:
            link.send_velocity(0.0, 0.0, 0.0)
            rk.wait()
            link.update()
            low = min(low, hgt())
            log_row(rows, n, kind, "hold", time.time() - t0, link, h0, alt0)
        held_s = time.time() - t_h
        if link.mode == "GUIDED":
            ok_back = tr.request_mode(link, pilot_mode, timeout=2)
            print(f"[hold] back to you in {link.mode}" + ("" if ok_back else " - mode change refused: TAKE OVER with CH8"))
        else:
            print(f"[hold] you took over ({link.mode}) - the script lets go")
    elif took_over:
        print(f"[stop] you took over ({link.mode}) - the script lets go")
    tr.beep(link, tr.TUNE_REFUSED)
    return {"n": n, "type": kind, "level": level, "mode": mode, "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "planned_turns": round(goal, 2), "turns": round(turned, 2), "impact_s": round(impact_s, 2), "hit_cut": cut,
            "peak_rate_rads": round(peak_rate, 2), "peak_tilt_deg": round(peak_tilt, 1), "height_before_m": round(h0, 2),
            "lowest_m": round(low, 2), "height_lost_m": round(h0 - low, 2), "b_handover_s": round(b_t, 2) if b_t is not None else None,
            "recovered": rec_s is not None and not rescue, "recovered_s": round(rec_s, 2) if rec_s is not None else None,
            "rescued": rescue, "ended": why, "hold_s": round(held_s, 1), "back_mode": link.mode, "volt": round(tr.volts(link), 2)}


def check(link):
    """Bench: never arms, never changes a mode or parameter. Switches, buzzer, the checks, link rates."""
    print("CHECK - never arms, never moves. Flip CH11 / CH12 / CH13 ON and OFF. Ctrl+C to stop.")
    imu, att = tr.Rate(), tr.Rate()
    was = {ch: tr.rc(link, ch) for ch in HIT_CH}
    t_print = 0.0
    while True:
        link.update()
        imu.feed(link.msgs.get("RAW_IMU"))
        att.feed(link.msgs.get("ATTITUDE_QUATERNION"))
        for ch, kind in HIT_CH.items():
            v = tr.rc(link, ch)
            if v > tr.CH_HIGH >= was[ch]:
                ok, out = checks(link)
                print(f"switch CH{ch} ON -> {kind} hit | buzzer: 3 short = 'hit starts', then 2 long = 'refused / ended'")
                tr.beep(link, tr.TUNE_READY)
                time.sleep(1.5)
                tr.beep(link, tr.TUNE_REFUSED)
                print("checks: " + ("ALL PASS" if ok else "would REFUSE") + " (on the bench 'flying' and 'height' are NO: right)")
                for nm, good, txt in out:
                    print(f"   {'ok ' if good else 'NO '} {nm:<30} {txt}")
            was[ch] = v
        if time.time() - t_print > 1.0:
            h = rel_height(link)
            print(f"CH11 {tr.rc(link, 11)} CH12 {tr.rc(link, 12)} CH13 {tr.rc(link, 13)} CH3 {tr.rc(link, 3)} | "
                  f"IMU {imu.hz:.0f} Hz ATT {att.hz:.0f} Hz | height {('%.1f m' % h) if h is not None else '-'} | "
                  f"{tr.volts(link):.2f} V | {link.mode}")
            t_print = time.time()
        time.sleep(0.02)


def save(base, results, rows, meta):
    with open(f"{base}.csv", "w", newline="") as f:
        keys = list(dict.fromkeys(k for e in results for k in e)) or ["n"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    with open(f"{base}_steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "type", "phase", "t_s", "tilt_deg", "rate_rads", "height_m", "hspeed_ms", "vdown_ms",
                    "p_rads", "q_rads", "r_rads"])
        w.writerows(rows)
    with open(f"{base}.json", "w") as f:
        json.dump(dict(meta, results=results), f, indent=2)
    for ext in (".csv", "_steps.csv", ".json"):
        fd = os.open(f"{base}{ext}", os.O_RDWR)       # RDWR: Windows (SITL on the PC) cannot fsync a read-only file
        os.fsync(fd)
        os.close(fd)


def main(args):
    if args.mode is None:
        txt = Path(tr.MODE_FILE).read_text().strip() if Path(tr.MODE_FILE).exists() else ""
        args.mode = txt if txt in ("attitude-rule", "ppo-policy") else "ppo-policy"
    signal.signal(signal.SIGTERM, tr.on_sigterm)
    net = None
    if args.mode == "ppo-policy" and not args.check:
        if not Path(tr.ACTOR).exists():
            raise SystemExit(f"No {tr.ACTOR}: run on the PC  py .\\export_ppo_onnx.py  and copy the .onnx to the Pi")
        net = tr.OnnxActor(tr.ACTOR)
        print(f"PPO (ONNX) loaded: {net.source}")
    link = tr.RealLink(args.conn, baud=args.baud, rate_hz=50)
    if args.check:
        t_end = time.time() + 10
        while time.time() < t_end and link.q is None:
            link.update()
            time.sleep(0.05)
    else:
        for _ in range(8):
            try:
                link.wait_state(timeout=15)
                break
            except RuntimeError:
                print("waiting for the EKF position (GPS fix) ...")
        else:
            raise SystemExit("No position from the EKF after 2 min - check GPS")
    for msg_id, hz in ((M.MAVLINK_MSG_ID_RAW_IMU, 50), (M.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 10),
                       (M.MAVLINK_MSG_ID_RC_CHANNELS, 20), (M.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10),
                       (M.MAVLINK_MSG_ID_SYS_STATUS, 2), (M.MAVLINK_MSG_ID_GPS_RAW_INT, 2),
                       (M.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 2)):
        link.request_rate(msg_id, hz)
    need = {"RAW_IMU", "EXTENDED_SYS_STATE", "RC_CHANNELS", "GLOBAL_POSITION_INT"} | set(tr.DATA_MSGS)
    t_end, t_ask = time.time() + 20, time.time()
    while time.time() < t_end and not need <= set(link.msgs):
        link.update()
        if time.time() - t_ask > 2.0:
            tr.ask_data_again(link)
            link.request_rate(M.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10)
            t_ask = time.time()
        time.sleep(0.05)
    if need - set(link.msgs):
        raise SystemExit(f"The FC does not send {', '.join(sorted(need - set(link.msgs)))} - check SERIAL2 (2 / 921)")
    if args.check:
        try:
            check(link)
        except KeyboardInterrupt:
            print("\ncheck stopped")
        return
    p = link.get_params(["GUID_OPTIONS", "MOT_THST_HOVER", ic.FF_NAME])
    hover = float(p.get("MOT_THST_HOVER", 0.149))
    guid, limits = tr.restore_left_values(link, int(p.get("GUID_OPTIONS", 0)), ic.read_limits(link))
    ff = p.get(ic.FF_NAME)
    if ff is not None and int(ff) == 0:
        raise SystemExit(f"STOP - {ic.FF_NAME} is 0 (left from a stopped run?) - set it back to 1 in Mission Planner")
    if not limits:
        raise SystemExit("Could not read the ATC_ACCEL limits - not hitting without them")
    rng = np.random.RandomState(int(time.time()))
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / f"hit_real_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    meta = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "mode": args.mode, "level": args.level,
            "hover": hover, "model": net.source if net else None, "version": "a (12 Nov 2026)", "option_b": True,
            "min_height_m": MIN_HEIGHT_M, "cut_height_m": CUT_HEIGHT_M, "rescue_m": RESCUE_M, "hold_s": HOLD_S,
            "hit_ch": HIT_CH, "limits": limits, "guid_options": guid, "rate_ff": ff}
    results, rows, n = [], [], 0
    was = {ch: tr.rc(link, ch) for ch in HIT_CH}
    t_print = 0.0
    print(f"Connected | {args.mode} + Option B | level {args.level} | hover thrust {hover:.3f} | limits {limits}")
    tr.beep(link, tr.TUNE_START[args.mode])
    print("Pi READY - take off yourself, climb to 25-30 m, CH8 POSHOLD, sticks middle, then flip ONE hit switch "
          "(CH11 flip / CH12 bird / CH13 spin). Ctrl+C to end.")
    try:
        while True:
            link.update()
            for ch, kind in HIT_CH.items():
                v = tr.rc(link, ch)
                if v > tr.CH_HIGH >= was[ch]:
                    ok, out = checks(link)
                    if not ok:
                        tr.beep(link, tr.TUNE_REFUSED)
                        print("REFUSED: " + "; ".join(f"{nm} ({txt})" for nm, good, txt in out if not good))
                    else:
                        e = one_hit(link, kind, args.level, args.mode, n + 1, rows, hover, guid, limits, ff, net, rng)
                        if e is not None:
                            n += 1
                            results.append(e)
                            save(base, results, rows, meta)
                            print(f"Hit {n} | {kind} {args.level} | {args.mode} | turns {e['turns']}/{e['planned_turns']} in "
                                  f"{e['impact_s']} s | peak rate {e['peak_rate_rads']} | height lost {e['height_lost_m']} m "
                                  f"(lowest {e['lowest_m']} m) | B hand-over {e['b_handover_s']} s | "
                                  f"{'RECOVERED ' + str(e['recovered_s']) + ' s' if e['recovered'] else 'NOT recovered'}"
                                  f"{' | RESCUED' if e['rescued'] else ''} | {e['ended']} | now {e['back_mode']}")
                            print("Next hit: switch OFF, check height, switch ON.")
                    v = tr.rc(link, ch)
                was[ch] = v
            if time.time() - t_print > 2.0:
                tr.ask_data_again(link)
                h = rel_height(link)
                fix, sats = tr.gps(link)
                print(f"WAIT | height {('%.1f m' % h) if h is not None else '-'} | {link.mode} {'ARMED' if link.armed else 'disarmed'} | "
                      f"CH3 {tr.rc(link, 3)} | CH11 {tr.rc(link, 11)} CH12 {tr.rc(link, 12)} CH13 {tr.rc(link, 13)} | "
                      f"GPS {fix} {sats} sats | {tr.volts(link):.2f} V")
                t_print = time.time()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        ic.send_params(link, {"GUID_OPTIONS": guid})
        ic.send_params(link, limits)
        if ff is not None:
            ic.send_params(link, {ic.FF_NAME: ff})
        save(base, results, rows, meta)
        print(f"Saved: {base}.csv / _steps.csv / .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None, choices=["attitude-rule", "ppo-policy"],
                    help="default: the word in mode.txt, else ppo-policy (with Option B)")
    ap.add_argument("--level", default="easy", choices=["easy", "medium", "hard"], help="hit size (Stage 3 levels)")
    ap.add_argument("--check", action="store_true", help="bench: switches, buzzer, checks - never arms, never moves")
    ap.add_argument("--conn", default="/dev/serial0", help="Pi UART (SITL test: tcp:127.0.0.1:5762)")
    ap.add_argument("--baud", type=int, default=921600)
    try:
        main(ap.parse_args())
    except KeyboardInterrupt:
        print("stopped (before the WAIT loop - no hits, nothing to save)")
