r"""
hit_real.py - IN-FLIGHT HIT TEST: you hover the 5-inch Q250, flip a switch, the Pi hits it and the PPO recovers it
=====================================================================================================
The real-drone twin of impact_check.py. SITL 13 Nov 2026 (b): PPO + Option B from 10 m - 21 hits, 0 ground contact.
Same Pi, wiring, model and link as throw_real.py (it imports its helpers - keep both files in the folder).
Version 12 Nov 2026 (a): first version. Recovery = PPO + Option B (PPO levels, height hold brakes, ArduPilot holds).
Version 13 Nov 2026 (b): RESCUE FIX + 10 m SETTINGS (section 8). (a) only ended the full-thrust rescue when 'calm'
(not moving up OR down) - full thrust makes it climb, so it never ended: SITL rescue at 10 m climbed to 243 m.
(b) switches full thrust OFF as soon as it is upright and climbing (tilt < 30, rising > 0.3 m/s); ArduPilot's
GUIDED hold (which knows the real hover) then holds the height. SITL: never above the start height after a rescue.

0. SETTINGS AT THE TOP (change here only)
------------------------------------------------------------------------------------------------------
    | Setting      | Value                | Meaning                                                      |
    |--------------|----------------------|--------------------------------------------------------------|
    | HIT_CH       | 11 flip, 12 bird,    | LEFT buttons -> hit type (ON above 1700)                     |
    |              | 13 spin              |                                                              |
    | LEVEL_CH     | 14 easy, 15 medium,  | RIGHT buttons -> hit size (d, 17 Nov). The last one pressed  |
    |              | 16 hard              | stays until you press another. Start = --level (easy)        |
    | MIN_HEIGHT_M | 8                    | m above home: refuse a hit below this (hover at 10 m)        |
    | CUT_HEIGHT_M | 5                    | m: stop the hit early if it could not brake above this       |
    | RESCUE_M     | 3                    | m: below this (braking included) the rule rescues at full    |
    |              |                      | thrust - the hit then counts as FAILED                       |
    | RETURN_MAX_S | 30                   | e: s to fly BACK to the hit point; then handed to you anyway |
    | RETURN_OK_M  | 1.0                  | e: m from the hit point that counts as 'back'                |
    | MIN_VOLT     | 14.4                 | V in the air under load (4S, 3.6 V per cell)                 |

1. HOW TO RUN  (Pi, folder Q250_Hardware_Throw_Fly)
------------------------------------------------------------------------------------------------------
    FIELD: AUTOSTART, no laptop (section 12). Battery in -> the Pi starts hit_real.py by itself ->
    start tune + 1 HIGH beep (= hit_real, level easy) -> fly. Printout: runs/hit_console.log.

    BY HAND (bench / SITL): stop BOTH services first - only ONE program may use the port:

    sudo systemctl stop throw hit

    tmux new -s hit

    cd ~/Q250_Hardware_Throw_Fly

    python3 hit_real.py --check                    (bench: link, the 6 buttons, buzzer - never arms, never moves)

    python3 hit_real.py                            (PPO + Option B; starts at easy, change level with the right buttons)

    python3 hit_real.py --level medium             (only sets the STARTING level - the right buttons still change it)

    python3 hit_real.py --mode attitude-rule       (the rule, for comparison - same hits)

    SITL test (WSL, simulator running): add --conn tcp:127.0.0.1:5762 --baud 115200 - the ONLY difference.
    After the session: the autostart you chose for the day starts again at the next power-up (section 12).

2. THE BUZZER  (same tunes as throw_real.py)
------------------------------------------------------------------------------------------------------
    | You hear          | Means                                            | Do                            |
    |-------------------|--------------------------------------------------|-------------------------------|
    | 1 long + 2 short  | Pi started, PPO (1 long + 1 short = rule)        | wait for the high beep(s)     |
    | THEN 1 HIGH beep  | ...and it is HIT_REAL (level easy). throw_real    | take off yourself, PosHold    |
    |                   | never plays this - no high beep = THROW service  |                               |
    | 3 short beeps     | checks passed - the hit starts NOW               | hands off the sticks (middle) |
    | 2 long beeps      | REFUSED (see printout) or hit ended / you took   | read the console later        |
    |                   | over                                             |                               |
    | 1 / 2 / 3 HIGH    | level set: easy / medium / hard (right buttons)  | then press a left hit button  |
    | short beeps       |                                                  |                               |

3. ONE HIT  (you fly; the Pi only acts between your switch and the hand-back)
------------------------------------------------------------------------------------------------------
    | # | Who      | Does                                                      | Script                       |
    |---|----------|-----------------------------------------------------------|------------------------------|
    | 1 | You      | take off, climb to 10 m, CH8 POSHOLD, sticks middle       | WAIT (prints height)         |
    | 2 | You      | RIGHT button = level (CH14 easy/CH15 medium/CH16 hard)    | 1/2/3 HIGH beeps, '[level]'  |
    | 2b| You      | then ONE LEFT button (CH11 flip / CH12 bird / CH13 spin)  | checks -> 3 short beeps      |
    | 3 | Pi       | GUIDED; hit = spin the drone N turns (rotation limits off)| "HIT flip easy 1.6 turns"    |
    | 4 | Pi       | recovery: PPO levels -> height hold brakes the fall        | 50 Hz, Option B              |
    |   |          |  (attitude-rule: the rule does both)                       |                              |
    | 5 | Pi       | e: FLIES BACK to the hit point (same spot, same height),    | "[return] flying back ..."   |
    |   |          |    max 2 m/s sideways, 1.5 m/s up/down, sits still 2 s     | "[return] back at the hit    |
    |   |          |    (not back in 30 s -> step 6 anyway, 2 long beeps)       |  point in 6.3 s"             |
    | 6 | Pi       | back to YOUR mode (POSHOLD) right there, 2 long beeps      | "back to you in POSHOLD"     |
    | 7 | You      | next hit: right button (level), left button (hit)          | result line, files saved     |
    | ! | You      | CH8 to any mode at ANY step = the script lets go at once   | "you took over"              |

4. CHECKS WHEN A HIT SWITCH GOES ON  (all must pass, else "REFUSED: ..." + 2 long beeps)
------------------------------------------------------------------------------------------------------
    | # | Check                  | Needed                        | Why                                      |
    |---|------------------------|-------------------------------|------------------------------------------|
    | 1 | armed, flying          | landed_state "in air"         | never hits a drone on the ground         |
    | 2 | your mode              | POSHOLD or LOITER             | the Pi hands back to this mode           |
    | 3 | height above home      | >= MIN_HEIGHT_M (8 m)         | hover at 10 m (section 8)                |
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
    | could not brake above 5 m       | hit stopped early ("cut"), recovery starts at once          |
    | would go below 3 m              | RESCUE: rule at full thrust until upright + not falling,    |
    |                                 | then GUIDED hold; the hit counts as FAILED (13 Nov b fix)   |
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
    Same hits in impact_check.py (12 per method, battery 15.3 V): PPO alone 5/12 (avg 14.0 m lost), PPO + Option B
    12/12 (avg 6.6 m, worst 14.7 m), attitude-rule 12/12 (avg 2.7 m).

8. VERSION b (13 Nov 2026): RESCUE FIX + 10 m SETTINGS  (your field rule: no 30 m; full story: hit_real guide)
------------------------------------------------------------------------------------------------------
    | Change            | a (12 Nov)                             | b (13 Nov)                               |
    |-------------------|----------------------------------------|------------------------------------------|
    | rescue ends when  | spin < 1.2, tilt < 30, |vertical| < 1  | tilt < 30 and climbing (> 0.3 m/s up)   |
    |                   | m/s -> never under full thrust (climbs)| -> full thrust off, ArduPilot GUIDED hold|
    | MIN / CUT / RESCUE| 20 / 12 / 6 m (for 30 m)               | 8 / 5 / 3 m (for 10 m)                   |
    SITL 13 Nov (SITL 1e001fa0, q250.json, battery 15.3 V, sensor 2 cm off), hover 10 m, PPO + Option B,
    THIS version b, levels easy + medium: 18 hits + 3 in the sim_pilot.py check = 21 hits:
    | Hit   | Height lost  | Lowest point | Result                                                         |
    |-------|--------------|--------------|----------------------------------------------------------------|
    | spin  | 3.3 - 4.4 m  | >= 6.0 m     | 7/7 recovered                                                  |
    | bird  | 3.8 - 6.1 m  | >= 4.0 m     | 3 recovered, 4 rescued                                         |
    | flip  | 3.6 - 6.1 m  | >= 4.3 m     | 5 recovered, 2 rescued                                         |
    | All   |              | >= 4.0 m     | 15 recovered, 6 rescued, 0 ground contact; after a rescue never |
    |       |              |              | more than 1.8 m above the start height (version a: 243 m)      |
    Rescue test (RESCUE_M raised to 6 m to force it): 4 rescues, all ended at 5.9 - 7.1 m, back to POSHOLD.
    Real drone at 10 m: CH13 spin (easy -> hard) first; bird / flip only when the real spin losses match the
    simulator (3-5 m); hard bird / flip came closest to the ground (3.2 m) - leave them out at 10 m.

9. VERSION c (16 Nov 2026): THE 14 NOV GUARDS, AND A 10 x 10 m FIELD
------------------------------------------------------------------------------------------------------
    Version b wrote GUID_OPTIONS / the rotation limits ONCE and never read them back - the same hole that
    caused the 14 Nov throw fly-away (throw_real version f). Guards A and B from throw_real version g are
    now here. C (READY sends 0) and D (motor watchdog) do not apply: there is no hand-held READY state and
    the drone is already flying on purpose.
    | Guard | What it does                                                   | What you see                  |
    |-------|----------------------------------------------------------------|-------------------------------|
    | A     | EVERY parameter write is repeated every 0.1 s until the FC      | [params] confirmed by the FC  |
    |       | echoes it (ParamKeeper); after every hit the script waits for   | ... or [params] NOT CONFIRMED |
    |       | the confirmation (3 s) before it lets go                        | + 2 long beeps                |
    | B     | At EVERY hit switch ON: read GUID_OPTIONS + the limits FROM THE | [fix] thrust switch / limits  |
    |       | FC, repair + read back if wrong, REFUSE the hit if it will not  | were wrong - fixed and read   |
    |       | confirm                                                         | back  /  REFUSED: ...         |
    Your field is 10 x 10 m (16 Nov), so two new numbers:
    | Number        | Value | Means                                                                        |
    |---------------|-------|------------------------------------------------------------------------------|
    | FIELD_R_M     | 5.0 m | the spot where you START the script is the middle of the field. A hit further  |
    |               |       | than this from it is REFUSED - fly back to the middle, then switch ON          |
    | DRIFT_MAX_M   | 4.0 m | drift from the HIT POINT. Past this the 10 s GUIDED hold is skipped and the    |
    |               |       | drone is handed back to your mode as soon as it is level, so you can fly back  |
    | DRIFT_WARN_M  | 2.5 m | only printed: "[drift] 2.8 m from the hit point"                               |
    The drone is NOT handed back while it is still tumbling: PPO recovers better than a mid-tumble mode
    change would. The drift limit shortens the hold, it does not interrupt the recovery.
    New columns in the .csv / .json: drift_m, drift_abort.
    Reference drift (real throws, 15 Nov): 0.45 - 6.6 m. A hit at 10 m has more momentum, so expect more.
    What version c does NOT change: the hit itself, the recovery, Option B, the rescue, the levels.

10. VERSION d (17 Nov 2026): LEVEL FROM THE RADIO  (Jumper Bumblebee: 3 left buttons + 3 right buttons)
------------------------------------------------------------------------------------------------------
    | Buttons    | Channels                       | Does                                                 |
    |------------|--------------------------------|------------------------------------------------------|
    | LEFT 3     | CH11 flip, CH12 bird, CH13 spin| start a hit (same as before)                         |
    | RIGHT 3    | CH14 easy, CH15 medium,        | set the hit size. 1 / 2 / 3 high beeps confirm it.   |
    |            | CH16 hard                      | It stays set until you press another right button.   |
    One hit = press a RIGHT button (hear the beeps), then a LEFT button.
    --level only chooses the level the script STARTS with. The right buttons are ignored during a hit.
    Every hit's level is saved in its own result line, so a session can mix levels.
    On the drone: RC14_OPTION, RC15_OPTION, RC16_OPTION must be 0 (do nothing) - check in Mission Planner.
    Also fixed: --check (bench) crashed in version c - it called the checks the old way. It now reads the
    thrust switch but never writes to the FC, as the bench test promises.

11. VERSION e (17 Nov 2026): IT FLIES BACK TO WHERE IT WAS HIT
------------------------------------------------------------------------------------------------------
    Before (a-d): after the recovery the drone held a 10 s GUIDED hold WHEREVER IT HAD SLID TO, then was handed
    back to you there - lower than before (PPO), or 6-8 m higher (after a rescue), and metres away. Nothing
    brought it back; FIELD_R_M (5 m) only refuses the NEXT hit, it never moves the drone.
    Now (e): the 10 s hold is replaced by a flight back to the exact spot of the hit (position AND height):
    | Number         | Value   | Means                                                                    |
    |----------------|---------|--------------------------------------------------------------------------|
    | RETURN_SPEED   | 2 m/s   | sideways speed back (the same law as impact_check.py fly_back)           |
    | RETURN_VZ      | 1.5 m/s | up / down speed back - so after a rescue it comes DOWN to the hit height |
    | RETURN_OK_M    | 1.0 m   | this close to the hit point, slower than 0.5 m/s and level = "back"      |
    | RETURN_STILL_S | 2 s     | sits still at the hit point this long, then POSHOLD = yours              |
    | RETURN_MAX_S   | 30 s    | not back by then -> handed to you anyway + 2 long beeps                  |
    Removed: HOLD_S (the 10 s hold) and DRIFT_MAX_M (it only shortened that hold). FIELD_R_M stays.
    New result columns: returned, return_s, back_err_m. Printout: "| back YES 6.3 s" or "| back NO".
    New _steps.csv columns (at the END, old columns unchanged): north_m, east_m = where it is vs the hit point.
    Order inside one hit is unchanged up to the end of the recovery: hit -> PPO -> height hold -> (rescue)
    -> thrust switch put back and CONFIRMED -> fly back -> your POSHOLD.
    e (18 Nov 2026) - four small fixes from the last review, same behaviour otherwise. The .json says
    "e (18 Nov 2026)"; the DATE tells it apart from "e (17 Nov 2026)" (same idea as throw_real g2):
    | Fix | 17 Nov                                        | 18 Nov                                        |
    |-----|-----------------------------------------------|-----------------------------------------------|
    | 1   | 'back' needed tilt < 10 deg - in wind the     | 'back' = within 1 m + slower than 0.5 m/s     |
    |     | drone leans to hold still -> 'back NO' at 30 s| (tilt no longer counts)                        |
    | 2   | 2 m/s per axis -> 2.8 m/s on a diagonal       | 2 m/s in ANY direction (as written above)     |
    | 3   | CH8 during the fly-back saved as 'recovered'  | saved as 'you took over (fly-back)'           |
    | 4   | link lost while you stop the script -> crash, | still saves the logs, prints [exit] ...        |
    |     | logs not saved                                |                                               |
    | 5   | start tune only (same as throw_real)          | start tune + 1/2/3 high beeps = the level, so  |
    |     |                                               | you can HEAR it is hit_real (autostart, s. 12)|

12. AUTOSTART FOR THE FIELD (18 Nov 2026) - like throw_real, no laptop, no tmux
------------------------------------------------------------------------------------------------------
    /etc/systemd/system/hit.service runs  python3 -u hit_real.py  at boot (Conflicts=throw.service:
    the two can never run together - one serial port). You choose the job of the day AT HOME:
    | Day        | Commands (Pi, at home)                                      | Start sound in the field      |
    |------------|-------------------------------------------------------------|-------------------------------|
    | HIT day    | sudo systemctl disable --now throw                          | 1 long + 2 short, then 1 HIGH |
    |            | sudo systemctl enable hit                                   |                               |
    | THROW day  | sudo systemctl disable --now hit                            | 1 long + 2 short only         |
    |            | sudo systemctl enable throw                                 |                               |
    | check      | systemctl is-enabled throw hit                              | one 'enabled', one 'disabled' |
    Power the drone up IN THE MIDDLE OF THE FIELD: the spot where the script starts = the centre for the
    5 m check. If it ever restarts (crash), the new spot is where the drone is then.
    Crash / no GPS yet -> systemd starts it again after 5 s. Printout: runs/hit_console.log (read at home:
    grep -a "Hit \|REFUSED\|level\|return" runs/hit_console.log | tail -n 40).
    systemctl stop hit = Ctrl+C: parameters put back + confirmed, files saved. Pulling the battery loses
    nothing either: the files are written after EVERY hit, and the parameters are confirmed after every hit.
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
LEVEL_CH = {14: "easy", 15: "medium", 16: "hard"}   # d (17 Nov): right-hand buttons on the Bumblebee
TUNE_LEVEL = {"easy": b"MFT240L8O7e", "medium": b"MFT240L8O7eP8e", "hard": b"MFT240L8O7eP8eP8e"}   # 1 / 2 / 3 high beeps
MIN_HEIGHT_M = 8.0         # m above home: refuse below (b: hover at 10 m - your field rule; SITL section 8)
CUT_HEIGHT_M = 5.0         # m: stop the hit early if braking could not stop above this (a: 12 for 30 m)
RESCUE_M = 3.0             # m: rescue (full-thrust rule) if it would pass this even braking (a: 6 for 30 m)
RESCUE_END_VZ = -0.3       # m/s (NED, - = up): rescue ends once upright and climbing; GUIDED hold takes over (b)
RETURN_MAX_S = 30.0        # e (17 Nov): s to fly back to the hit point; not back by then -> handed back to you anyway
RETURN_SPEED = 2.0         # e: m/s sideways while flying back   (same as impact_check.py fly_back)
RETURN_VZ = 1.5            # e: m/s up / down while flying back  (same as fly_back)
RETURN_GAIN = 0.8          # e: speed command = 0.8 x distance, capped by the two above (same as fly_back)
RETURN_OK_M = 1.0          # e: m from the hit point = 'back' (fly_back uses 0.5 in SITL; real GPS needs ~1)
RETURN_STILL_S = 2.0       # e: s it must sit still at the hit point before it is handed back to you
RECOVER_MAX_S = 8.0        # s: not calm by then -> GUIDED hold anyway
MIN_VOLT = 14.4            # V in the air under load (4S)
FIELD_R_M = 5.0            # c (16 Nov): m. Half of your 10 x 10 m field. A hit further than this from the
                           #   START POINT (where the script was started) is REFUSED - fly back to the middle first.
DRIFT_WARN_M = 2.5         # c: m - printed once as a warning, nothing else happens (e: it flies back anyway)
PILOT_MODES = ("POSHOLD", "LOITER")
STICK_MID = (1400, 1600)   # throttle stick in the middle (PosHold holds height)
HZ = 50.0
OUT = Path("runs") / "hit_real"


def rel_height(link):
    """Height above home (m) from the FC (GLOBAL_POSITION_INT.relative_alt), None if not received."""
    m = link.msgs.get("GLOBAL_POSITION_INT")
    return None if m is None else m.relative_alt / 1000.0


def flat_dist(a, b):
    """c: horizontal distance (m) between two positions, height ignored."""
    return float(np.linalg.norm(np.asarray(a)[0:2] - np.asarray(b)[0:2]))


def switch_and_limits(link, keeper, guid, limits):
    """c (guard B), at every hit switch ON: read GUID_OPTIONS + the rotation limits FROM THE FC.
    Thrust bit left ON or a limit left at 0 -> put back, wait for the confirmation, read again.
    Returns (ok, text). Not ok = REFUSED, no hit. Same idea as check 10 in throw_real.py."""
    names = ["GUID_OPTIONS"] + list(limits)
    got = link.get_params(names, timeout=2)
    good = lambda g: (len(g) == len(names) and not int(g["GUID_OPTIONS"]) & ic.GUID_THRUST_BIT
                      and all(abs(g[k]) > 1e-6 for k in limits))
    if not good(got) and len(got) == len(names):
        keeper.set({"GUID_OPTIONS": guid, **limits})
        keeper.flush()
        got = link.get_params(names, timeout=2)
        if good(got):
            print(f"[fix] thrust switch / limits were wrong - fixed and read back: {got}")
    if len(got) < len(names):
        return False, "NO ANSWER from the FC"
    return good(got), (f"GUID_OPTIONS {got['GUID_OPTIONS']:.0f}"
                       + "".join(f", {k} {got[k]:.0f}" for k in limits if abs(got[k]) < 1e-6))


def confirm_hit(link, keeper):
    """c (guard A): after the script lets go, wait until the FC confirms the put-back values."""
    names = keeper.pending()
    if keeper.flush():
        if names:
            print("[params] confirmed by the FC: GUID_OPTIONS 0 (thrust switch OFF) + rotation limits")
        return True
    tr.beep(link, tr.TUNE_REFUSED)
    print(f"[params] NOT CONFIRMED by the FC: {', '.join(keeper.pending())} - do NOT hit again. "
          "Switch OFF; the next switch ON checks and fixes it (guard B)")
    return False


def checks(link, keeper, guid, limits, home, repair=True):
    """repair=False (bench --check): read the thrust switch only, never write to the FC."""
    fix, sats = tr.gps(link)
    h = rel_height(link)
    if repair:
        sw_ok, sw_txt = switch_and_limits(link, keeper, guid, limits)  # c (guard B)
    else:
        got = link.get_params(["GUID_OPTIONS"], timeout=2)
        sw_ok = "GUID_OPTIONS" in got and not int(got["GUID_OPTIONS"]) & ic.GUID_THRUST_BIT
        sw_txt = f"GUID_OPTIONS {got.get('GUID_OPTIONS', 'NO ANSWER')} (read only)"
    d = flat_dist(link.pos, home) if link.pos is not None and home is not None else None   # d: bench has no position
    out = [("armed, flying", link.armed and tr.landed_state(link) == 2,
            f"{'armed' if link.armed else 'DISARMED'}, {'in air' if tr.landed_state(link) == 2 else 'ON GROUND'}"),
           ("your mode POSHOLD / LOITER", link.mode in PILOT_MODES, link.mode),
           (f"height >= {MIN_HEIGHT_M:.0f} m above home", h is not None and h >= MIN_HEIGHT_M, f"{h:.1f} m" if h is not None else "no data"),
           ("throttle stick middle", STICK_MID[0] <= tr.rc(link, 3) <= STICK_MID[1], f"CH3 {tr.rc(link, 3)}"),
           ("GPS 3D fix + EKF GPS", fix >= 3 and sats >= tr.MIN_SATS and tr.ekf_gps(link), f"fix {fix}, {sats} sats, EKF {'yes' if tr.ekf_gps(link) else 'NO'}"),
           (f"battery >= {MIN_VOLT} V", tr.volts(link) >= MIN_VOLT, f"{tr.volts(link):.2f} V"),
           ("thrust switch OFF + limits", sw_ok, sw_txt),                      # c (guard B)
           (f"within {FIELD_R_M:.0f} m of the start point", d is not None and d <= FIELD_R_M, f"{d:.1f} m" if d is not None else "no position")]   # c (10 x 10 m field)
    return all(c[1] for c in out), out


_SPOT = None                                                     # e: the current hit point, for north_m / east_m


def log_row(rows, n, kind, phase, t, link, h0, alt0):
    off = (link.pos[0:2] - _SPOT[0:2]) if _SPOT is not None else (0.0, 0.0)   # e: where it is vs the hit point
    rows.append([n, kind, phase, round(t, 3), round(link.tilt_deg, 1), round(float(np.linalg.norm(link.rates)), 2),
                 round(h0 + link.alt - alt0, 2), round(float(np.linalg.norm(link.vel[0:2])), 2), round(float(link.vel[2]), 2)]
                + [round(float(v), 2) for v in link.rates] + [round(float(off[0]), 2), round(float(off[1]), 2)])


def one_hit(link, kind, level, mode, n, rows, hover, guid, limits, ff, net, rng, keeper):
    """Hit (lead hit, rotation limits off) -> recovery (PPO + Option B, or attitude-rule) -> GUIDED hold -> back.
    c (16 Nov): every parameter write goes through the keeper and is confirmed (guard A); drift from the hit
    point is measured. e (17 Nov): after the recovery it FLIES BACK to the hit point (position AND height),
    then hands back to you there."""
    pilot_mode, h0, alt0 = link.mode, rel_height(link), link.alt
    global _SPOT
    spot = link.pos.copy()                                       # c: the hit point; drift is measured from here
    _SPOT = spot                                                 # e: step log records north_m / east_m from it
    drift = 0.0
    said = 0.0                                                   # c: largest drift already printed about
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
        keeper.set({k: 0 for k in limits})                        # rotation limits off for the hit (c: confirmed)
    if ff is not None:
        keeper.set({ic.FF_NAME: 0})
    try:
        # ---- the hit
        while turned < goal and time.time() - t0 < ic.CAP * planned:
            ic.send_hit(link, (p, q, r))
            rk.wait()
            link.update()
            keeper.service()                                      # c (guard A)
            drift = max(drift, flat_dist(link.pos, spot))
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
            keeper.set({ic.FF_NAME: ff})                          # your value back BEFORE the recovery (c: confirmed)
        # ---- the recovery
        t_rec, b_t, rec_s, rescue, held, phase = time.time(), None, None, False, None, "ppo" if mode == "ppo-policy" else "rule"
        rescued = False                                      # b: True once a rescue happened (the hit = FAILED)
        last = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if not took_over:
            keeper.set({"GUID_OPTIONS": int(guid) | ic.GUID_THRUST_BIT})   # thrust = real thrust (c: confirmed)
        while not took_over and time.time() - t_rec < RECOVER_MAX_S:
            rate, tilt, h, vz = float(np.linalg.norm(link.rates)), link.tilt_deg, hgt(), float(link.vel[2])
            if not rescue and (not rescued or vz > tr.VZ_OK) and h - max(0.0, vz) ** 2 / (2 * ic.RESCUE_BRAKE) < RESCUE_M:
                rescue, rescued, phase = True, True, "rescue"
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
            keeper.service()                                      # c (guard A)
            low = min(low, hgt())
            d_now = flat_dist(link.pos, spot)
            if d_now > DRIFT_WARN_M > said:                        # c: printed once
                said = DRIFT_WARN_M
                print(f"[drift] {d_now:.1f} m from the hit point - it flies back once it is level")
            drift = max(drift, d_now)
            peak_rate = max(peak_rate, float(np.linalg.norm(link.rates)))
            log_row(rows, n, kind, phase, time.time() - t0, link, h0, alt0)
            if link.mode != "GUIDED" or not link.armed:
                took_over = True
                break
            if rescue and link.tilt_deg < ic.TILT_OK and float(link.vel[2]) < RESCUE_END_VZ:
                print("[rescue] fall stopped - ArduPilot GUIDED hold")   # b: full thrust OFF as soon as it climbs
                break                                                   #    (a: kept full thrust -> SITL 243 m)
            calm = (float(np.linalg.norm(link.rates)) < ic.RATE_OK and link.tilt_deg < ic.TILT_OK
                    and abs(float(link.vel[2])) < tr.VZ_OK)
            held = (held or time.time()) if calm else None
            if held and time.time() - held >= ic.HELD_S:
                rec_s = held - t_rec
                break
        why = "you took over" if took_over else ("rescued" if rescued else ("recovered" if rec_s is not None else f"not calm in {RECOVER_MAX_S:.0f} s"))
    finally:
        keeper.set({"GUID_OPTIONS": int(guid), **(limits or {}), **({ic.FF_NAME: ff} if ff is not None else {})})
        confirm_hit(link, keeper)                                 # c (guard A): the FC must echo them back
    # ---- e (17 Nov): fly back to the EXACT spot it was hit (position and height), then hand back to you there
    ret_s, back_err, returned = 0.0, None, False
    if not took_over and link.armed and link.mode == "GUIDED":
        print(f"[return] flying back to the hit point: {flat_dist(link.pos, spot):.1f} m away, "
              f"height {hgt():.1f} m -> {h0:.1f} m")
        t_r, still = time.time(), None
        while time.time() - t_r < RETURN_MAX_S and link.mode == "GUIDED" and link.armed:
            err = spot - link.pos
            vh = RETURN_GAIN * err[0:2]                            # e-18Nov: cap the SPEED, not each axis
            sh = float(np.linalg.norm(vh))                         #   (17 Nov: 2.8 m/s on a diagonal)
            if sh > RETURN_SPEED:
                vh = vh * (RETURN_SPEED / sh)
            link.send_velocity(float(vh[0]), float(vh[1]), float(np.clip(RETURN_GAIN * err[2], -RETURN_VZ, RETURN_VZ)))
            rk.wait()
            link.update()
            keeper.service()
            low = min(low, hgt())
            drift = max(drift, flat_dist(link.pos, spot))
            log_row(rows, n, kind, "return", time.time() - t0, link, h0, alt0)
            here = (float(np.linalg.norm(spot - link.pos)) < RETURN_OK_M and float(np.linalg.norm(link.vel)) < 0.5)
            #   e-18Nov: no tilt test any more - in wind the drone LEANS to stay put (17 Nov: tilt < 10 deg)
            still = (still or time.time()) if here else None
            if still and time.time() - still >= RETURN_STILL_S:
                returned = True
                break
        ret_s = time.time() - t_r
        back_err = float(np.linalg.norm(spot - link.pos))
        if link.mode == "GUIDED":
            if returned:
                print(f"[return] back at the hit point in {ret_s:.1f} s ({back_err:.1f} m off)")
            else:
                tr.beep(link, tr.TUNE_REFUSED)
                print(f"[return] NOT back after {RETURN_MAX_S:.0f} s ({back_err:.1f} m off) - handing it to you anyway")
            ok_back = tr.request_mode(link, pilot_mode, timeout=2)
            if not ok_back:
                ok_land = tr.request_mode(link, "LAND", timeout=2)
                print(f"[return] {pilot_mode} refused - " + ("LAND" if ok_land else "LAND REFUSED: TAKE OVER with CH8"))
            print(f"[return] back to you in {link.mode}")
        else:
            why = "you took over (fly-back)"                       # e-18Nov: was saved as 'recovered'
            print(f"[return] you took over ({link.mode}) - the script lets go")
    elif took_over:
        print(f"[stop] you took over ({link.mode}) - the script lets go")
    tr.beep(link, tr.TUNE_REFUSED)
    return {"n": n, "type": kind, "level": level, "mode": mode, "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "planned_turns": round(goal, 2), "turns": round(turned, 2), "impact_s": round(impact_s, 2), "hit_cut": cut,
            "peak_rate_rads": round(peak_rate, 2), "peak_tilt_deg": round(peak_tilt, 1), "height_before_m": round(h0, 2),
            "lowest_m": round(low, 2), "height_lost_m": round(h0 - low, 2), "b_handover_s": round(b_t, 2) if b_t is not None else None,
            "recovered": rec_s is not None and not rescued, "recovered_s": round(rec_s, 2) if rec_s is not None else None,
            "rescued": rescued, "ended": why, "back_mode": link.mode,
            "drift_m": round(drift, 2),                            # c: furthest it got from the hit point
            "returned": returned, "return_s": round(ret_s, 1),     # e: flew back to the hit point?
            "back_err_m": round(back_err, 2) if back_err is not None else None,
            "volt": round(tr.volts(link), 2)}


def check(link):
    """Bench: never arms, never changes a mode or parameter. Switches, buzzer, the checks, link rates."""
    print("CHECK - never arms, never moves, never writes to the FC. Press each LEFT button (CH11/12/13) and each "
          "RIGHT button (CH14/15/16). Ctrl+C to stop.")
    imu, att = tr.Rate(), tr.Rate()
    link.update()
    home = link.pos.copy() if link.pos is not None else None
    was = {ch: tr.rc(link, ch) for ch in list(HIT_CH) + list(LEVEL_CH)}
    t_print = 0.0
    while True:
        link.update()
        imu.feed(link.msgs.get("RAW_IMU"))
        att.feed(link.msgs.get("ATTITUDE_QUATERNION"))
        for ch, kind in HIT_CH.items():
            v = tr.rc(link, ch)
            if v > tr.CH_HIGH >= was[ch]:
                ok, out = checks(link, None, 0, {}, home, repair=False)
                print(f"switch CH{ch} ON -> {kind} hit | buzzer: 3 short = 'hit starts', then 2 long = 'refused / ended'")
                tr.beep(link, tr.TUNE_READY)
                time.sleep(1.5)
                tr.beep(link, tr.TUNE_REFUSED)
                print("checks: " + ("ALL PASS" if ok else "would REFUSE") + " (on the bench 'flying' and 'height' are NO: right)")
                for nm, good, txt in out:
                    print(f"   {'ok ' if good else 'NO '} {nm:<30} {txt}")
            was[ch] = v
        for ch, lv in LEVEL_CH.items():
            v = tr.rc(link, ch)
            if v > tr.CH_HIGH >= was[ch]:
                print(f"button CH{ch} -> level {lv} | buzzer: {list(LEVEL_CH.values()).index(lv) + 1} high beep(s)")
                tr.beep(link, TUNE_LEVEL[lv])
            was[ch] = v
        if time.time() - t_print > 1.0:
            h = rel_height(link)
            print(f"CH11 {tr.rc(link, 11)} CH12 {tr.rc(link, 12)} CH13 {tr.rc(link, 13)} | "
                  f"CH14 {tr.rc(link, 14)} CH15 {tr.rc(link, 15)} CH16 {tr.rc(link, 16)} | CH3 {tr.rc(link, 3)} | "
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
                    "p_rads", "q_rads", "r_rads", "north_m", "east_m"])
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
    keeper = tr.ParamKeeper(link)                                # c (guard A)
    link.update()
    home = link.pos.copy()                                       # c: the middle of your field = where you start me
    rng = np.random.RandomState(int(time.time()))
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / f"hit_real_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    meta = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "mode": args.mode, "level": args.level,
            "hover": hover, "model": net.source if net else None, "version": "e (18 Nov 2026)", "option_b": True,
            "min_height_m": MIN_HEIGHT_M, "cut_height_m": CUT_HEIGHT_M, "rescue_m": RESCUE_M,
            "field_r_m": FIELD_R_M, "return_max_s": RETURN_MAX_S, "return_ok_m": RETURN_OK_M,
            "hit_ch": HIT_CH, "level_ch": LEVEL_CH, "limits": limits, "guid_options": guid, "rate_ff": ff}
    results, rows, n = [], [], 0
    level = args.level                                           # d: the right buttons change it
    was = {ch: tr.rc(link, ch) for ch in list(HIT_CH) + list(LEVEL_CH)}
    t_print = 0.0
    print(f"Connected | {args.mode} + Option B | level {level} (right buttons: CH14 easy / CH15 medium / CH16 hard) "
          f"| hover thrust {hover:.3f} | limits {limits}")
    tr.beep(link, tr.TUNE_START[args.mode])
    time.sleep(1.6)                                              # e-18Nov autostart: let the start tune finish, then
    tr.beep(link, TUNE_LEVEL[level])                             #   1/2/3 HIGH beeps = this is hit_real (not throw_real)
    print(f"[start] start tune + {list(LEVEL_CH.values()).index(level) + 1} high beep(s) = hit_real ready, level {level}")
    print(f"Pi READY - the spot where you started me is the middle of the field (a hit further than "
          f"{FIELD_R_M:.0f} m from it is refused).")
    print("Take off yourself, climb to 10 m, CH8 POSHOLD, sticks middle. RIGHT button = level (1/2/3 high beeps), "
          "then LEFT button = hit (CH11 flip / CH12 bird / CH13 spin). Ctrl+C to end.")
    try:
        while True:
            link.update()
            for ch, lv in LEVEL_CH.items():                      # d: right buttons set the level
                v = tr.rc(link, ch)
                if v > tr.CH_HIGH >= was[ch]:
                    level = lv
                    tr.beep(link, TUNE_LEVEL[lv])
                    print(f"[level] {lv} (CH{ch}) - {list(LEVEL_CH.values()).index(lv) + 1} high beep(s)")
                was[ch] = v
            for ch, kind in HIT_CH.items():
                v = tr.rc(link, ch)
                if v > tr.CH_HIGH >= was[ch]:
                    ok, out = checks(link, keeper, guid, limits, home)
                    if not ok:
                        tr.beep(link, tr.TUNE_REFUSED)
                        print("REFUSED: " + "; ".join(f"{nm} ({txt})" for nm, good, txt in out if not good))
                    else:
                        e = one_hit(link, kind, level, args.mode, n + 1, rows, hover, guid, limits, ff, net, rng, keeper)
                        if e is not None:
                            n += 1
                            results.append(e)
                            save(base, results, rows, meta)
                            print(f"Hit {n} | {kind} {level} | {args.mode} | turns {e['turns']}/{e['planned_turns']} in "
                                  f"{e['impact_s']} s | peak rate {e['peak_rate_rads']} | height lost {e['height_lost_m']} m "
                                  f"(lowest {e['lowest_m']} m) | B hand-over {e['b_handover_s']} s | "
                                  f"{'RECOVERED ' + str(e['recovered_s']) + ' s' if e['recovered'] else 'NOT recovered'}"
                                  f"{' | RESCUED' if e['rescued'] else ''} | drift {e['drift_m']} m"
                                  f" | back {'YES ' + str(e['return_s']) + ' s' if e['returned'] else 'NO'} | {e['ended']} | now {e['back_mode']}")
                            print(f"Next hit: switch OFF, check height, then a right button for the level (now {level}) and a left button.")
                    v = tr.rc(link, ch)
                    for lc in LEVEL_CH:                          # d: a right button pressed DURING the hit is ignored
                        was[lc] = tr.rc(link, lc)
                was[ch] = v
            if time.time() - t_print > 2.0:
                tr.ask_data_again(link)
                h = rel_height(link)
                fix, sats = tr.gps(link)
                print(f"WAIT | level {level} | height {('%.1f m' % h) if h is not None else '-'} | {link.mode} {'ARMED' if link.armed else 'disarmed'} | "
                      f"CH3 {tr.rc(link, 3)} | CH11 {tr.rc(link, 11)} CH12 {tr.rc(link, 12)} CH13 {tr.rc(link, 13)} | "
                      f"GPS {fix} {sats} sats | {tr.volts(link):.2f} V")
                t_print = time.time()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        try:
            keeper.set({"GUID_OPTIONS": guid, **limits, **({ic.FF_NAME: ff} if ff is not None else {})})
            confirm_hit(link, keeper)                             # c (guard A): never leave the FC wrong on exit
        except Exception as e:                                    # e: link gone on exit -> still save the logs
            print(f"[exit] could not confirm settings ({e}) - check GUID_OPTIONS in Mission Planner before the next flight")
        save(base, results, rows, meta)
        print(f"Saved: {base}.csv / _steps.csv / .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None, choices=["attitude-rule", "ppo-policy"],
                    help="default: the word in mode.txt, else ppo-policy (with Option B)")
    ap.add_argument("--level", default="easy", choices=["easy", "medium", "hard"], help="STARTING hit size - the right buttons (CH14/15/16) change it in flight")
    ap.add_argument("--check", action="store_true", help="bench: switches, buzzer, checks - never arms, never moves")
    ap.add_argument("--conn", default="/dev/serial0", help="Pi UART (SITL test: tcp:127.0.0.1:5762)")
    ap.add_argument("--baud", type=int, default=921600)
    try:
        main(ap.parse_args())
    except KeyboardInterrupt:
        print("stopped (before the WAIT loop - no hits, nothing to save)")
