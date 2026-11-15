r"""
throw_real.py - REAL THROW: you throw the 5-inch Q250 by hand, the Raspberry Pi Zero 2 W catches it
=====================================================================================================
Same detection + catch as throw_check.py (SITL, 8 Nov 2026: 27/27 caught). Pi Zero 2 W (aarch64, Debian 12,
Python 3.11): onnxruntime + numpy + pymavlink (no torch). Full steps: "Throw & Fly 5-inch Guide", Part C.
Version 8 Nov 2026 (b): the THROTTLE STICK STAYS LOW THE WHOLE TIME, the buzzer tells you READY / REFUSED.
Version 8 Nov 2026 (c): NO fake EKF origin is sent any more - the real GPS sets it (section 8, RealLink).
Version 8 Nov 2026 (d): mode back to ALT_HOLD after cancel / auto-disarm / landing (arm again at once), hold
10 s before LAND (time to take over), GPS + battery data asked again until it arrives, quiet stop at start.
Version 12 Nov 2026 (e): FAST-SPIN THROWS ARE NO LONGER MISSED (real throw 6, 11 Nov) - section 9:
Fix 1 free-fall limit = 0.35 g + the spin's push (SPIN_R_M 0.03 m); Fix 2 no detection 0.5 m below the hand.
Version 12 Nov 2026 (f): OPTION B + B2 - the PPO only levels the drone, the height hold brakes the fall, ArduPilot
brakes the drift (section 10). Fast flips in the simulator: PPO 2/13 -> 8/8 caught.

0. SETTINGS AT THE TOP (change here only)
------------------------------------------------------------------------------------------------------
    | Setting   | Value | Meaning                                                              |
    |-----------|-------|----------------------------------------------------------------------|
    | THROW_CH  | 5     | throw switch (2-position, RC5_OPTION 0): ON = Pi watches for a throw |
    | HOLD_S    | 10    | s position hold (GUIDED) after the catch, then LAND (was 3, 8 Nov d) |
    | MAX_UP_M  | 5     | m above the hand: hand over at once                                  |

1. HOW TO RUN  (folder Q250_Hardware_Throw_Fly - PC and Pi use the same name)
------------------------------------------------------------------------------------------------------
    PC, once: make the ONNX PPO file -> separate script, see export_ppo_onnx.py.

    Pi (from the PC: ssh jlukas@10.160.112.138), ALWAYS inside tmux - a dropped Wi-Fi link must not kill
    the script in the air:

    tmux new -s throw

    cd ~/Q250_Hardware_Throw_Fly

    python3 throw_real.py --check                  (step 0: link OK, switch OK, buzzer test - never arms)

    python3 throw_real.py                          (Round 1 PPO - the default, your choice 8 Nov)

    python3 throw_real.py --mode attitude-rule     (the rule, for comparison)

    Detach: Ctrl+B then D.  Back: tmux attach -t throw.  End: Ctrl+C (drone disarmed).
    SITL test (WSL, simulator running): add --conn tcp:127.0.0.1:5762 - the ONLY difference to the drone.

2. THE BUZZER  (the drone's own buzzer, played by the Pi through ArduPilot - MAVLink PLAY_TUNE)
------------------------------------------------------------------------------------------------------
    | You hear                  | Means                                              | Do                    |
    |---------------------------|----------------------------------------------------|-----------------------|
    | 3 short beeps             | READY: Pi has the drone (GUIDED_NOGPS), props still| throw (no time limit) |
    | 2 long beeps              | REFUSED / cancelled / disarmed while ready         | CH5 OFF, read the Pi  |
    | 1 long + 1 short          | Pi started, link + GPS OK, mode attitude-rule      | you may arm           |
    | 1 long + 2 short          | Pi started, link + GPS OK, mode ppo-policy         | you may arm           |
    | ArduPilot's own tunes     | arm, disarm (as always)                            | -                     |
    --check plays both patterns when you flip CH5 ON, so you learn them and know the buzzer works.

3. ONE THROW  (throttle stick LOW the whole time - props can never start in the hand)
------------------------------------------------------------------------------------------------------
    | # | Who      | Does                                                    | Script / you hear              |
    |---|----------|---------------------------------------------------------|--------------------------------|
    | 1 | Pilot    | CH6 middle BEFORE the battery; CH8 ALT_HOLD; stick LOW  | (WAIT: prints status)          |
    | 2 | Pilot    | CH9 arm. Props still. Stick stays LOW                   | ArduPilot arming tone          |
    | 3 | Pilot    | CH5 ON                                                  | 9 checks -> GUIDED_NOGPS ->    |
    |   |          |                                                         | 3 SHORT BEEPS = READY          |
    | 4 | Thrower  | toss up and away (READY waits until CH5 OFF)            | push > 1.5 g, then 3 readings  |
    |   |          |                                                         | in a row < limit (section 9)   |
    |   |          |                                                         | -> motors start (~0.35 s)      |
    | 5 | Pi       | catch: attitude-rule or PPO until level and calm        | rotation limits off, 50 Hz     |
    | 6 | ArduPilot| GUIDED position hold 10 s -> LAND -> disarm (stick ignored)| (requests the modes, logs)  |
    | 7 | Pilot    | take over ANY time (catch, hold, LAND): stick to MIDDLE,| "you took over (POSHOLD) - the |
    |   |          | then CH8 to POSHOLD (or Loiter; never Stabilize)        |  script lets go"               |
    | 8 | Anyone   | pick it up ONLY after it is disarmed; CH5 OFF; CH9 OFF  | mode back to ALT_HOLD by itself|
    |   |          |                                                         | -> arm again at once (8 Nov d) |
    Why the stick stays LOW: ALT_HOLD on the ground with the stick at middle or above prepares take-off and
    spins the props (ArduPilot get_alt_hold_state; SITL PWM 1000 -> 1153). GUIDED_NOGPS and GUIDED ignore the
    stick, and your firmware (4.6.0-dev 1e001fa0, mode_guided.cpp line 966) starts the motors on the Pi's
    take-off command even with the stick at the bottom. NO 10 s limit (corrected 8 Nov d): your CH9 arms with
    AirMode (RC9_OPTION 154) and ArduPilot then never auto-disarms (radio.cpp set_throttle_zero_flag:
    armed_with_airmode_switch -> throttle never "zero" -> DISARM_DELAY not used; SITL with CH9 154: READY
    held 25 s, props still). READY lasts until you flip CH5 OFF (or CH9 / CH7). The earlier "10 s" came from
    a SITL test that armed by MAVLink, not by your switch. After you land it yourself: disarm with CH9.

4. CHECKS WHEN CH5 GOES ON  (all must pass, else "REFUSED: ..." + 2 long beeps, and nothing happens)
------------------------------------------------------------------------------------------------------
    | # | Check               | Needed                          | Why                                         |
    |---|---------------------|---------------------------------|---------------------------------------------|
    | 1 | armed, on the ground| you armed it (CH9), not flying  | the script never takes over a flying drone  |
    | 2 | MOT_SPIN_ARM        | 0                               | props still in your hand                    |
    | 3 | MOT_SPOOL_TIME      | <= 0.3 (throw day 0.25)         | motors on ~0.35 s (0.5: ~0.6 s, 23 m drift) |
    | 4 | RC5_OPTION          | 0                               | CH5 must not start AutoTune                 |
    | 5 | CH6                 | middle (GPS set)                | position hold after the catch               |
    | 6 | GPS                 | 3D fix, >= 5 satellites         | (your choice; EKF needs 6 to START on GPS)  |
    | 7 | EKF GPS position    | EKF_STATUS "absolute position"  | = "EKF3 IMU0 is using GPS"; GUIDED needs it |
    | 8 | CH3 throttle stick  | LOW (below 1300)                | ALT_HOLD + stick middle = props spin        |
    | 9 | battery             | >= 14.9 V (3.72 V per cell)     | enough thrust for the catch (was 15.2, 8 Nov)|
    At start the script also puts back GUID_OPTIONS / ATC_ACCEL_* if a stopped run left test values saved.

5. TAKE OVER / ABORT  (any time)
------------------------------------------------------------------------------------------------------
    | You do / happens              | Result                                                          |
    |-------------------------------|-----------------------------------------------------------------|
    | CH5 OFF while READY           | Pi disarms (2 long beeps)                                       |
    | CH7 (disarm)                  | motors off at once (ArduPilot); script puts parameters back     |
    | stick MIDDLE, CH8 -> POSHOLD  | ArduPilot changes mode -> script stops commanding at once. YOU  |
    |  (catch, hold or LAND)        | fly it (full control); the script never changes your mode      |
    | CH5 OFF in the air            | NOT a take-over: GUIDED hold 10 s now, then LAND                |
    | climbs > 5 m above the hand   | GUIDED hold 10 s now, then LAND                                 |
    | not caught 3 s after motors on| GUIDED hold 10 s now (ArduPilot levels it), then LAND           |
    | cancel / auto-disarm / landed | Pi's mode left on the FC -> script sets ALT_HOLD (disarmed only)|
    | GUIDED refused (no position)  | LAND straight away (lands level, 0.2 m/s near the ground)       |
    | Pi stops in the air           | after GUID_TIMEOUT 3 s ArduPilot levels + holds height: CH8     |

6. OUTPUTS  (runs/throw_real/, written AND flushed to the SD card after EVERY throw)
------------------------------------------------------------------------------------------------------
    throw_real_<date>.csv        one line per throw (motors_on_s, caught_s, recovered_s, low_m, drift_m, ...)
    throw_real_<date>_steps.csv  every 50 Hz step: phase, tilt, rate, height above the hand, speed, accel, CH5
    throw_real_<date>.json       settings + every throw.   + the FC's own .bin log (Mission Planner)

7. FIELD USE WITHOUT A LAPTOP  (autostart when the battery goes in - systemd service, guide C2 6)
------------------------------------------------------------------------------------------------------
    | What                  | How                                                                        |
    |-----------------------|----------------------------------------------------------------------------|
    | starts by itself      | /etc/systemd/system/throw.service runs  python3 -u throw_real.py  at boot  |
    | which mode            | the word in mode.txt (attitude-rule / ppo-policy), set at home             |
    | "Pi is ready"         | 1 long + 1 short (rule) / 1 long + 2 short (PPO), ~1 min after power-up    |
    | crash / no GPS yet    | systemd starts it again after 5 s                                          |
    | GPS / battery data    | asked again every 2 s until it arrives (a lost request once gave 0 sats, 0 V)|
    | printout              | runs/console.log (read it at home)                                         |
    | --check by hand       | sudo systemctl stop throw   first (only ONE program may use the port)      |

8. SAME AS THE SIMULATOR  (change one = change both)
------------------------------------------------------------------------------------------------------
    free fall < limit x 3 (0.35 g + spin push, section 9 - throw_check.py e has the same rule), spool rule, attitude-rule / PPO step (impact_check.py), caught and calm rules,
    hover thrust = MOT_THST_HOVER from the FC (0.149). New here: push > 1.5 g first, CH5, the checks,
    buzzer, no SIM_SHOVE, no RC override, GUIDED hold instead of LOITER (does not need the stick).
    NOT the same (8 Nov fix c): ap_link.APLink sends the SITL default EKF origin (Canberra, 584 m) at connect -
    the simulator needs it without GPS. On the real drone it is WRONG: ArduPilot keeps the FIRST origin until
    reboot (AP_NavEKF3 setOrigin), so a Pi that starts before GPS lock fixed home + compass table to Australia.
    | Before (b)                                  | Now (c)                                            |
    |---------------------------------------------|----------------------------------------------------|
    | log: "EKF3 IMU0 origin set" + "Field        | no such line from the Pi; the FC sets the origin   |
    |  Elevation Set: 584m" right after connect   |  itself from GPS (as in a normal flight)           |
    | start beep could come without GPS (indoors) | start beep only after a real GPS position          |

9. VERSION e (12 Nov 2026): WHY THROW 6 FELL, AND THE 2 FIXES  (full story: "Why Throw 6 Crashed" guide)
------------------------------------------------------------------------------------------------------
    Real field day 11 Nov (.bin 2026-11-11_16-29-50): throws 1-5 caught, throw 6 fell. Throw 6 spun 15 rad/s
    (2.4 turns/s; throws 1-5: 4-8 rad/s). The FC's sensor sits ~2 cm from the spin centre, so the spin pressed
    it outward: in free fall it read 0.35-0.64 g instead of ~0 g -> never below 0.35 g -> motors stayed OFF.
    Then the crash (6.3 g) looked like a push and the bounce like free fall -> "detected" ON THE GROUND.
    The simulator never showed it: its sensor sits exactly at the centre (SIM_IMU_POS 0).
    Spin push (g) = spin x spin x distance / 9.81.  Distance from the log = reading x 9.81 / (spin x spin):
    throw 6, 25 readings in the air -> 1.6-2.6 cm (middle 1.9 cm). SPIN_R_M rounds UP to 3 cm.
    | Rule                          | Before (d)               | Now (e)                                        |
    |-------------------------------|--------------------------|------------------------------------------------|
    | free fall = below             | 0.35 g                   | 0.35 g + spin x spin x 0.03 / 9.81             |
    |   in the hand (spin ~0)       | 0.35 g                   | 0.35 g (same)                                  |
    |   throw 4 (4.5 rad/s)         | 0.35 g                   | 0.41 g                                         |
    |   throw 6 (15 rad/s)          | 0.35 g -> MISSED         | 1.04 g -> detected 0.08 s after release        |
    | 0.5 m or more below the hand  | detected (motors on the  | "[guard] ... not a throw", motors stay OFF     |
    |                               |  ground, upside down)    |                                                |
    Unchanged: push > 1.5 g first, 0.8 s window after the push, 3 readings in a row, everything after detection.
    Replay of the 11 Nov log: throws 1-5 detected at the same moments; throw 6 at +0.08 s (motors ~0.3 s later,
    ~1 s before the ground); no detection in the hand during any READY. Simulator test with the sensor 2 cm
    off-centre (SIM_IMU_POS_Z 0.02): see the guide, part "Simulator test".
    New printouts:
    | Line                                                                    | Means                         |
    |-------------------------------------------------------------------------|-------------------------------|
    | Throw detected - spooling motors  (0.49 g < limit 1.07 g at spin 15.3)  | the reading, limit and spin   |
    | [guard] -1.1 m below the hand - not a throw (crash / dropped), motors.. | Fix 2 blocked a false start   |
    Simulator, sensor 2 cm off-centre (SIM_IMU_POS_Z 0.02), fast flip ~15.6 rad/s, THIS script with your switches:
    | Version | Mode          | Release found | Result                          |
    |---------|---------------|---------------|---------------------------------|
    | d       | ppo-policy    | 0.34 s late   | CRASHED (ground at 5.9 m/s)     |
    | d       | attitude-rule | 0.34 s late   | recovered, lowest +1.64 m       |
    | e       | ppo-policy    | 0.06 s        | recovered, lowest +0.87 m       |
    | e       | attitude-rule | 0.06 s        | recovered, lowest +1.37 m       |
    throw_check.py, same flip, release found in time: attitude-rule 6/6 caught, Round 1 PPO only 1/8 (also
    with the sensor at the centre) -> at throw-6 spins the PPO's CATCH is weak, not the detection. Real drone: gentle throws first;
    for fast flips use attitude-rule (mode.txt) until the PPO handles them.

10. VERSION f (12 Nov 2026): OPTION B + B2  (full story: "Q250 Recovery Option B" guide)
------------------------------------------------------------------------------------------------------
    Found: once level, the Round 1 PPO asks only ~25 % of hover thrust (it trained 30 m up and lost ~15 m per
    recovery). The hand-over to the height hold needed "falling < 1 m/s", so a drone that levelled while
    already falling (fast flips) stayed with the PPO and fell. Gentle throws levelled while still RISING.
    | Hand-over                 | Before (e)                                | Now (f)                              |
    |---------------------------|-------------------------------------------|--------------------------------------|
    | PPO -> height hold        | spin < 1.2 rad/s, tilt < 30 deg,          | spin < 1.2 rad/s, tilt < 30 deg      |
    |  ("caught")               |  falling < 1 m/s                          |  (falling allowed: B)                |
    | height hold -> ArduPilot  | + drift < 3 m/s + |vertical| < 1 m/s,     | + |vertical| < 1 m/s, 0.5 s          |
    |  GUIDED hold ("recovered")|  0.5 s                                    |  (ArduPilot brakes the drift: B2)    |
    Simulator (SITL 1e001fa0, q250.json, sensor 2 cm off-centre), flip-fast ~15.6 rad/s:
    | Script                        | Result                                                            |
    |-------------------------------|-------------------------------------------------------------------|
    | throw_check, PPO (e)          | 2/13 caught                                                       |
    | throw_check, PPO + B          | 8/8 caught, lowest 1.4-2.1 m                                      |
    | throw_real (your switches)    | e+B only: caught, but the 2.4 s height hold sank to the ground;    |
    |                               | e+B+B2 (= f): 2/2 PPO + 1/1 rule, lowest 1.3-1.8 m                |
    | up / roll / pitch             | unchanged (all caught, same times)                               |
    New column meaning: caught_s = level + spin stopped; recovered_s = level + not falling (drift may go on).
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
from ap_link import APLink, RateKeeper, M


class RealLink(APLink):
    """APLink for the REAL drone: never sends an EKF origin (section 8). The real GPS sets it."""

    def set_ekf_origin(self, *args, **kwargs):
        pass

THROW_CH = 5               # throw switch channel (2-position, RC5_OPTION 0)
HOLD_S = 10.0              # s GUIDED position hold after the catch, then LAND (was 3; 8 Nov d: time to take over)
MAX_UP_M = 5.0             # m above the hand: hand over at once (7 Nov plan)
HZ = 50.0
FREEFALL_G = 0.35          # accelerometer below this = free fall (released)   - same as throw_check
SPIN_R_M = 0.03            # m: sensor distance from the spin centre allowed for (12 Nov e, Fix 1). Throw 6 log:
                           # 1.6-2.6 cm, rounded UP (too small = fast spins missed; bigger = harmless)
GROUND_GUARD_M = -0.5      # m: no throw is detected this far BELOW the hand (12 Nov e, Fix 2: crash != throw)
FREEFALL_N = 3             # ... for this many samples in a row (60 ms)         - same as throw_check
PUSH_G = 1.5               # g: the throw push must come first (7 Nov plan)
PUSH_TO_FREE_S = 0.8       # s: free fall must start within this time after the push
SPOOL_MAX_S = 1.0          # s max for the motors to reach full range           - same as throw_check
RECOVER_MAX_S = 3.0        # s after motors on: not caught by then = hand over (ArduPilot levels it)
STALE_S = 0.5              # s without attitude data from the FC = hand over
HOLD_K = 0.4               # x hover per m/s: height hold after the catch      - same as throw_check
VZ_OK = 1.0                # m/s: calm also needs this vertical speed          - same as throw_check
STOP_V = 0.2               # m/s: finished braking after the hand-over          - same as throw_check
LAND_WAIT_S = 60.0         # s for LAND to land and disarm
PI_MODES = ("GUIDED_NOGPS", "GUIDED", "LAND")   # modes only the script uses
DATA_MSGS = {"GPS_RAW_INT": 2, "SYS_STATUS": 2, "EKF_STATUS_REPORT": 2}   # needed by the checks (Hz)
LANDED_ON_GROUND = 1       # EXTENDED_SYS_STATE.landed_state
CH_HIGH, CH_LOW = 1700, 1300        # throw switch: ON above, OFF below
CH6_MID = (1300, 1700)     # CH6 middle = EKF source set 2 = GPS (RC6_OPTION 90)
CH3_LOW = 1300             # throttle stick LOW the whole time (ALT_HOLD on the ground: middle = props spin)
MIN_SATS = 5                # your choice (8 Nov). EKF3 still needs 6 to START using GPS (EK3_GPS_CHECK 31)
MIN_VOLT = 14.9            # V, 4S: 3.72 V per cell, at rest (your choice 8 Nov; was 15.2). Hover 0.149 = big thrust margin
REAL_LIMITS = {"ATC_ACCEL_R_MAX": 282459.0, "ATC_ACCEL_P_MAX": 282459.0,   # your drone, 7 Nov 2026 param file
               "ATC_ACCEL_Y_MAX": 31500.0}                                  # (used only if one was left at 0)
ACTOR = "models/round1_final.onnx"         # export_ppo_onnx.py (PC) writes it; --mode ppo-policy reads it
OUT = Path("runs") / "throw_real"


# ------------------------------------------------------------------ PPO on numpy (no torch on the Pi)
class OnnxActor:
    """The PPO (normalise + actor) as ONNX, run with onnxruntime. Same actions as the PPO (export_ppo_onnx.py checks)."""

    def __init__(self, path=ACTOR):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1                         # Pi Zero 2 W: one core is plenty, others free
        self.sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.source = self.sess.get_modelmeta().custom_metadata_map.get("source", Path(path).name)

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return self.sess.run(None, {"obs": x})[0][0].astype(np.float32)


def ppo_step(net, link, hover, alt0, last):
    """Same step as impact_check.policy_action (observation, yaw brake, action -> attitude), numpy network."""
    a = net(ic.bc_observation(link, alt0, last))
    q, thr, _ = ic.action_to_attitude(a, link, hover)
    return q, thr, a


# ------------------------------------------------------------------ small helpers (as throw_check)
def accel_g(link):
    m = link.msgs.get("RAW_IMU")
    return None if m is None else math.sqrt(m.xacc ** 2 + m.yacc ** 2 + m.zacc ** 2) / 1000.0


def spin_rads(link):
    """Spin from the SAME message as the accelerometer (RAW_IMU gyro, ArduPilot sends mrad/s)."""
    m = link.msgs.get("RAW_IMU")
    return 0.0 if m is None else math.sqrt(m.xgyro ** 2 + m.ygyro ** 2 + m.zgyro ** 2) / 1000.0


def freefall_limit(w):
    """Fix 1 (12 Nov e): free-fall limit in g. A spinning sensor that is off the spin centre is pressed outward
    by w*w*r (centripetal), so a falling drone does NOT read 0 g. Real throw 6 (11 Nov): 15 rad/s, 0.36-0.64 g ->
    the old fixed 0.35 g missed it. 0 rad/s -> 0.35 g (as before); 4.5 rad/s -> 0.41 g; 15 rad/s -> 1.04 g."""
    return FREEFALL_G + w * w * SPIN_R_M / 9.80665


def landed_state(link):
    m = link.msgs.get("EXTENDED_SYS_STATE")
    return int(m.landed_state) if m is not None else 0


def level_q(link):
    R = ic.quat_to_R(link.q)
    return ic.euler_quat(0.0, 0.0, math.atan2(R[1, 0], R[0, 0]))


def height_hold(link, hover):
    up = max(0.5, 1.0 - 2.0 * (link.q[1] ** 2 + link.q[2] ** 2))
    thr = hover * (1.0 + HOLD_K * float(link.vel[2])) / up
    return level_q(link), float(np.clip(thr, 0.3 * hover, 1.0))


def rc(link, ch):
    m = link.msgs.get("RC_CHANNELS")
    return int(getattr(m, f"chan{ch}_raw")) if m is not None else 0


def volts(link):
    m = link.msgs.get("SYS_STATUS")
    return m.voltage_battery / 1000.0 if m is not None and m.voltage_battery != 65535 else 0.0


def gps(link):
    m = link.msgs.get("GPS_RAW_INT")
    return (int(m.fix_type), int(m.satellites_visible)) if m is not None else (0, 0)


TUNE_READY = b"MFT240L8O6cP8cP8c"      # 3 short beeps
TUNE_REFUSED = b"MFT100L2O4cP4c"        # 2 long beeps
TUNE_START = {"attitude-rule": b"MFT120L2O5cP8L8c",       # Pi ready, rule: 1 long + 1 short
              "ppo-policy": b"MFT120L2O5cP8L8cP8c"}       # Pi ready, PPO:  1 long + 2 short
MODE_FILE = "mode.txt"                  # field use without a laptop: the mode the Pi starts with (section 7)


def beep(link, tune):
    """Play a tune on the drone's buzzer (MAVLink PLAY_TUNE -> ArduPilot AP_Notify / ToneAlarm)."""
    with link.io_lock:
        link.m.mav.play_tune_send(link.m.target_system, link.m.target_component, tune)


def ekf_gps(link):
    """EKF_STATUS_REPORT flag EKF_POS_HORIZ_ABS (16): the EKF has an absolute (GPS) position -> GUIDED / LOITER work."""
    m = link.msgs.get("EKF_STATUS_REPORT")
    return m is not None and bool(int(m.flags) & 16)


def disarm(link):
    link.m.arducopter_disarm()
    t_end = time.time() + 2
    while time.time() < t_end and link.armed:
        link.update()
        time.sleep(0.05)


def request_mode(link, mode, timeout=1.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        link.m.set_mode(mode)
        time.sleep(0.1)
        link.update()
        if link.mode == mode:
            return True
    return False


class Rate:
    """Counts how often a message arrives (Hz) - the Pi must see RAW_IMU / ATTITUDE at ~50 Hz."""

    def __init__(self):
        self.n, self.t0, self.last_id, self.hz = 0, time.time(), None, 0.0

    def feed(self, msg):
        if msg is not None and id(msg) != self.last_id:
            self.n, self.last_id = self.n + 1, id(msg)
        if time.time() - self.t0 >= 1.0:
            self.hz, self.n, self.t0 = self.n / (time.time() - self.t0), 0, time.time()


# ------------------------------------------------------------------ the checks
def checks(link, p):
    """Section 4. Returns (all passed, list of (name, ok, value))."""
    fix, sats = gps(link)
    on_ground = landed_state(link) == LANDED_ON_GROUND
    out = [("armed by you, on the ground", link.armed and on_ground,
            f"{'armed' if link.armed else 'DISARMED - arm with CH9'}, {'on ground' if on_ground else 'IN THE AIR'}"),
           ("MOT_SPIN_ARM 0", abs(p.get("MOT_SPIN_ARM", 1.0)) < 1e-6, f"{p.get('MOT_SPIN_ARM')}"),
           ("MOT_SPOOL_TIME <= 0.3", p.get("MOT_SPOOL_TIME", 0.5) <= 0.3 + 1e-6, f"{p.get('MOT_SPOOL_TIME')}"),
           (f"RC{THROW_CH}_OPTION 0", int(p.get(f"RC{THROW_CH}_OPTION", -1)) == 0, f"{p.get(f'RC{THROW_CH}_OPTION')}"),
           ("CH6 middle (GPS)", CH6_MID[0] <= rc(link, 6) <= CH6_MID[1], f"CH6 {rc(link, 6)}"),
           (f"GPS 3D fix, >= {MIN_SATS} sats", fix >= 3 and sats >= MIN_SATS, f"fix {fix}, {sats} sats"),
           ("EKF using GPS position", ekf_gps(link), "yes" if ekf_gps(link) else "NO - wait for 'EKF3 IMU0 is using GPS'"),
           ("throttle stick LOW", 900 < rc(link, 3) < CH3_LOW, f"CH3 {rc(link, 3)}"),
           (f"battery >= {MIN_VOLT} V", volts(link) >= MIN_VOLT, f"{volts(link):.2f} V")]
    return all(c[1] for c in out), out


def restore_left_values(link, guid, limits):
    """A run stopped mid-throw can leave the thrust bit / zero rotation limits SAVED on the FC -> put back."""
    fix = {}
    if guid & ic.GUID_THRUST_BIT:
        fix["GUID_OPTIONS"] = guid & ~ic.GUID_THRUST_BIT
    for k, v in limits.items():
        if abs(v) < 1e-6:
            if k not in REAL_LIMITS:
                raise SystemExit(f"STOP - {k} = 0 left from a stopped run: set it back in Mission Planner")
            fix[k] = REAL_LIMITS[k]
    if fix:
        print(f"[fix] values left from a stopped run - restored: {fix}")
        ic.send_params(link, fix)
    return int(fix.get("GUID_OPTIONS", guid)), {k: fix.get(k, v) for k, v in limits.items()}


def ask_data_again(link, max_age=3.0):
    """Change 2 (8 Nov d): a SET_MESSAGE_INTERVAL request can be lost on the serial line -> GPS / battery / EKF
    never arrive (0 sats, 0.00 V). Ask again for every one that is missing or older than max_age s."""
    now = time.time()
    for name, hz in DATA_MSGS.items():
        m = link.msgs.get(name)
        if m is None or now - getattr(m, "_timestamp", 0.0) > max_age:
            link.request_rate(getattr(M, f"MAVLINK_MSG_ID_{name}"), hz)


def back_to_alt_hold(link):
    """Change 1 (8 Nov d): after a cancel, auto-disarm or LAND the FC stays in the Pi's mode and ArduPilot will not
    arm there ("GUIDED_NOGPS mode not armable"). Only when DISARMED and still in a Pi mode: set ALT_HOLD.
    A mode you chose yourself (take-over) is never touched."""
    link.update()
    if link.mode not in PI_MODES:
        return
    t_end = time.time() + 3.0
    while link.armed and time.time() < t_end:
        link.update()
        time.sleep(0.05)
    if link.armed or link.mode not in PI_MODES:
        return
    if request_mode(link, "ALT_HOLD", timeout=3):
        print("mode back to ALT_HOLD - arm again with CH9 when ready")
    else:
        print(f"mode stays {link.mode}: flick CH8 to another mode and back to ALT_HOLD before arming")


# ------------------------------------------------------------------ one throw
def log_row(rows, n, mode, phase, t, link, h0, spot, g):
    hs = float(np.linalg.norm(link.vel[0:2]))
    rows.append([n, mode, phase, round(t, 3), round(link.tilt_deg, 1), round(float(np.linalg.norm(link.rates)), 2),
                 round(link.alt - h0, 2), round(hs, 2), round(float(np.linalg.norm(link.pos[0:2] - spot[0:2])), 2),
                 round(g, 3) if g is not None else "", landed_state(link), rc(link, THROW_CH)]
                + [round(float(v), 2) for v in link.rates])


def hold_land(link, n, mode, rows, t0_log, h0, spot, rk, why):
    """After the catch (or an abort): GUIDED position hold HOLD_S (no stick needed) -> LAND -> until disarmed.
    No position (GUIDED refused): LAND at once. Stops at once if the pilot changes the mode himself."""
    hold = "GUIDED" if request_mode(link, "GUIDED") else None
    if hold is None and not request_mode(link, "LAND"):
        print(f"[handover] GUIDED and LAND refused ({why}) - mode {link.mode}: TAKE OVER (stick middle, CH8 LOITER)")
        return {"hold_mode": link.mode}
    print(f"[handover] {'GUIDED hold ' + str(int(HOLD_S)) + ' s, then LAND' if hold else 'no position: LAND now'} ({why})")
    point, stop_pt, t_stop, wander, note = link.pos.copy(), None, None, 0.0, None
    t0 = time.time()
    while link.armed:
        if hold and link.mode == "GUIDED":
            link.send_velocity(0.0, 0.0, 0.0)                    # stay here (stick ignored)
        rk.wait()
        link.update()
        log_row(rows, n, mode, "hold" if link.mode == "GUIDED" else "land", time.time() - t0_log, link, h0, spot,
                accel_g(link))
        if stop_pt is None and float(np.linalg.norm(link.vel[0:2])) < STOP_V:
            stop_pt, t_stop = link.pos.copy(), time.time() - t0
        if stop_pt is not None and link.mode == "GUIDED":
            wander = max(wander, float(np.linalg.norm(link.pos[0:2] - stop_pt[0:2])))
        if link.mode == "GUIDED" and time.time() - t0 >= HOLD_S:
            request_mode(link, "LAND")
        elif link.mode not in ("GUIDED", "LAND"):
            note = f"you took over ({link.mode})"
            print(f"[handover] {note} - the script lets go")
            break
        if time.time() - t0 > HOLD_S + LAND_WAIT_S:
            note = "LAND did not disarm in 60 s - land it yourself"
            print(f"[handover] {note}")
            break
    if not link.armed:
        print("[link] landed and disarmed - pick it up now, then CH5 OFF, CH9 OFF")
    return {"hold_mode": hold or "LAND", "brake_s": round(t_stop, 2) if t_stop is not None else None,
            "brake_m": round(float(np.linalg.norm(stop_pt[0:2] - point[0:2])), 2) if stop_pt is not None else None,
            "wander_m": round(wander, 2) if stop_pt is not None else None,
            "landed_disarmed": not link.armed, "note": note}


def one_throw(link, mode, n, rows, hover, guid, limits, net):
    """READY (armed by the pilot, GUIDED_NOGPS, props still) -> throw -> catch -> GUIDED hold 3 s -> LAND.
    Returns the result dict, or None if no throw happened."""
    rk = RateKeeper(HZ)
    if not request_mode(link, "GUIDED_NOGPS", timeout=3):
        beep(link, TUNE_REFUSED)
        print(f"REFUSED by ArduPilot: GUIDED_NOGPS not accepted (mode {link.mode})")
        return None
    link.update()
    spot, h0 = link.pos.copy(), link.alt                     # the hand = height 0
    beep(link, TUNE_READY)
    print(f"READY - THROW NOW (3 short beeps, props still, stick stays LOW). CH{THROW_CH} OFF = cancel")
    t_ready = time.time()
    phase, t_push, n_ff = "ready", None, 0
    t_rel = t_motors = t_upright = t_caught = t_rec = held = spool_t0 = None
    alt_rel, last = None, np.array([0.0, 0.0, 1.0], dtype=np.float32)
    apex, low, peak_rate, tilted = 0.0, None, 0.0, False
    why, limits_off = None, False
    t_att, att_msg = time.time(), None
    while True:
        if phase == "ready":
            ic.send_attitude(link, level_q(link), 0.5)               # climb-rate mode, 0.5 = stay on the ground
        elif phase == "spool":
            ic.send_attitude(link, level_q(link), 1.0)               # take-off = spool the motors up
        elif phase == "caught":
            ic.send_attitude(link, *height_hold(link, hover))
        elif mode == "attitude-rule":
            ic.send_attitude(link, *ic.attitude_rule(link, hover))
        else:
            q, thr, last = ppo_step(net, link, hover, alt_rel, last)
            ic.send_attitude(link, q, thr)
        rk.wait()
        link.update()
        t, g = time.time(), accel_g(link)
        m_att = link.msgs.get("ATTITUDE_QUATERNION")
        if m_att is not att_msg:
            att_msg, t_att = m_att, t
        h = link.alt - h0
        log_row(rows, n, mode, phase, t - t_ready, link, h0, spot, g)
        rate = float(np.linalg.norm(link.rates))
        if t_rel is not None:
            peak_rate, apex = max(peak_rate, rate), max(apex, h)
            low = h if low is None else min(low, h)
            tilted = tilted or link.tilt_deg > 30
        # ---- READY: wait for push, then free fall
        if phase == "ready":
            if not link.armed:
                beep(link, TUNE_REFUSED)
                print("Disarmed while READY (CH9 or CH7) - back to WAIT")
                return None
            if link.mode != "GUIDED_NOGPS":
                beep(link, TUNE_REFUSED)
                print(f"Mode changed to {link.mode} while READY - back to WAIT")
                return None
            if rc(link, THROW_CH) < CH_LOW:
                disarm(link)
                beep(link, TUNE_REFUSED)
                print(f"CH{THROW_CH} OFF - cancelled, disarmed. Back to WAIT")
                return None
            if g is not None and g > PUSH_G:
                t_push = t
            if t_push is not None and t - t_push < PUSH_TO_FREE_S and g is not None:
                w = spin_rads(link)
                lim = freefall_limit(w)                                  # Fix 1: 0.35 g + the spin's push
                if h < GROUND_GUARD_M:                                   # Fix 2: below the hand = on the ground
                    if n_ff or g < lim:
                        print(f"[guard] {h:.1f} m below the hand - not a throw (crash / dropped), motors stay OFF")
                    n_ff, t_push = 0, None
                    continue
                n_ff = n_ff + 1 if g < lim else 0
                if n_ff >= FREEFALL_N:
                    t_rel, alt_rel = t - (FREEFALL_N - 1) / HZ, link.alt
                    phase, spool_t0 = "spool", t
                    print(f"Throw detected - spooling motors  ({g:.2f} g < limit {lim:.2f} g at spin {w:.1f} rad/s)")
            else:
                n_ff = 0
            continue
        # ---- in the air: aborts first
        if not link.armed:
            why = "disarmed (CH7)"
            break
        if link.mode != "GUIDED_NOGPS":
            why = f"you changed the mode to {link.mode}"
            break
        if rc(link, THROW_CH) < CH_LOW:
            why = f"CH{THROW_CH} OFF"
            break
        if h > MAX_UP_M:
            why = f"more than {MAX_UP_M:.0f} m above the hand"
            break
        if t - t_att > STALE_S:
            why = "no attitude data from the FC"
            break
        # ---- spool -> recover -> caught -> calm (same rules as throw_check)
        if phase == "spool" and (landed_state(link) != LANDED_ON_GROUND or t - spool_t0 > SPOOL_MAX_S):
            t_motors = t
            if limits:
                ic.send_params(link, {k: 0 for k in limits})          # rotation limits off (as the SITL test)
                limits_off = True
            ic.send_params(link, {"GUID_OPTIONS": int(guid) | ic.GUID_THRUST_BIT})   # thrust = real thrust
            phase = "recover"
        if t_upright is None and tilted and t_motors is not None and link.tilt_deg < ic.TILT_OK:
            t_upright = t
        if phase == "recover" and rate < ic.RATE_OK and link.tilt_deg < ic.TILT_OK:   # f (Option B): falling allowed
            if limits_off:
                ic.send_params(link, limits)
                limits_off = False
            phase, t_caught = "caught", t
        if phase == "recover" and t_motors is not None and t - t_motors > RECOVER_MAX_S:
            why = f"not caught in {RECOVER_MAX_S:.0f} s"
            break
        if t_motors is not None:
            ok = (rate < ic.RATE_OK and link.tilt_deg < ic.TILT_OK
                  and abs(float(link.vel[2])) < VZ_OK)               # f (B2): drift is braked by ArduPilot GUIDED
            held = (held or t) if ok else None
            if held and t - held >= ic.HELD_S:
                t_rec, why = held, "recovered"
                break
    # ---- the script's control ends: parameters back FIRST, then GUIDED hold 3 s -> LAND
    if limits_off:
        ic.send_params(link, limits)
    after = {}
    if link.armed and link.mode == "GUIDED_NOGPS":
        after = hold_land(link, n, mode, rows, t_ready, h0, spot, rk, why)
    ic.send_params(link, {"GUID_OPTIONS": int(guid)})
    if why != "recovered":
        print(f"[stop] {why}")
    rel = (lambda x: round(x - t_rel, 2) if (x is not None and t_rel is not None) else None)
    info = {"n": n, "mode": mode, "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "detected": t_motors is not None, "motors_on_s": rel(t_motors),
            "upright_s": (rel(t_upright) if tilted else 0.0) if t_motors else None, "caught_s": rel(t_caught),
            "recovered": t_rec is not None, "recovered_s": rel(t_rec), "ended": why,
            "apex_m": round(apex, 2), "low_m": round(low, 2) if low is not None else None,
            "peak_rate_rads": round(peak_rate, 2),
            "drift_m": round(float(np.linalg.norm(link.pos[0:2] - spot[0:2])), 2),
            "volt": round(volts(link), 2), "hover_used": round(float(hover), 3)}
    info.update(after)
    return info


# ------------------------------------------------------------------ --check (ladder step 0)
def check(link, p, net):
    """Step 0: "link OK, switch OK". Never arms, never commands. Also shows the throw detection and the checks."""
    print("CHECK - never arms, never changes a mode or parameter. Ctrl+C to stop.\n"
          f"Flip CH{THROW_CH} OFF and ON -> 'switch OK' + the check list. With CH{THROW_CH} ON, toss it ~20 cm onto a "
          "pillow -> 'throw detected (check only)'. Walking / shaking -> nothing.")
    imu, att = Rate(), Rate()
    t_push, n_ff, ch_was = None, 0, rc(link, THROW_CH)
    saw_low = saw_high = link_ok = False
    g_min, g_max, t_print = 9.0, 0.0, 0.0
    last = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rk = RateKeeper(HZ)
    while True:
        rk.wait()
        link.update()
        imu.feed(link.msgs.get("RAW_IMU"))
        att.feed(link.msgs.get("ATTITUDE_QUATERNION"))
        g, t, ch = accel_g(link), time.time(), rc(link, THROW_CH)
        if g is not None:
            g_min, g_max = min(g_min, g), max(g_max, g)
        if not link_ok and imu.hz >= 45 and att.hz >= 45:
            link_ok = True
            print(f"link OK  (IMU {imu.hz:.0f} Hz, ATT {att.hz:.0f} Hz)")
        saw_low, saw_high = saw_low or ch < CH_LOW, saw_high or ch > CH_HIGH
        if ch > CH_HIGH >= ch_was:
            if saw_low:
                print(f"switch OK  (CH{THROW_CH} OFF {CH_LOW}- / ON {CH_HIGH}+, now {ch})")
            ok, out = checks(link, p)
            print("buzzer test: 3 SHORT beeps = READY ...")
            beep(link, TUNE_READY)
            time.sleep(2.0)
            print("            ... 2 LONG beeps = REFUSED")
            beep(link, TUNE_REFUSED)
            print("checks: " + ("ALL PASS" if ok else "would REFUSE") + "  (arming is your CH9 in a real run)")
            for name, good, txt in out:
                print(f"   {'ok ' if good else 'NO '} {name:<30} {txt}")
        ch_was = ch
        if ch > CH_HIGH and g is not None:
            if g > PUSH_G:
                t_push = t
            lim = freefall_limit(spin_rads(link))                        # same rule as a real throw (Fix 1)
            n_ff = n_ff + 1 if (g < lim and t_push is not None and t - t_push < PUSH_TO_FREE_S) else 0
            if n_ff == FREEFALL_N:
                print(f"throw detected (check only - motors NOT started)  push {g_max:.2f} g, free fall {g_min:.2f} g"
                      f" (limit {lim:.2f} g)")
        if t - t_print >= 1.0:
            ms = None
            if net is not None and link.pos is not None and link.vel is not None:
                t_ns = time.perf_counter()
                net(ic.bc_observation(link, link.alt, last))
                ms = (time.perf_counter() - t_ns) * 1000
            fix, sats = gps(link)
            print(f"CH{THROW_CH} {ch} CH6 {rc(link, 6)} CH3 {rc(link, 3)} | GPS fix {fix} {sats} sats | {volts(link):.2f} V | "
                  f"IMU {imu.hz:.0f} Hz ATT {att.hz:.0f} Hz | g {g_min:.2f}-{g_max:.2f} | tilt {link.tilt_deg:.0f}"
                  + ((f" | PPO {ms:.1f} ms" if ms is not None else " | PPO - (no EKF position yet)") if net is not None else ""))
            g_min, g_max, t_print = 9.0, 0.0, t


# ------------------------------------------------------------------ main
def save(base, results, rows, meta):
    with open(f"{base}.csv", "w", newline="") as f:
        keys = list(dict.fromkeys(k for e in results for k in e)) or ["n"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    with open(f"{base}_steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "mode", "phase", "t_s", "tilt_deg", "rate_rads", "height_above_hand_m", "hspeed_ms",
                    "dist_from_hand_m", "accel_g", "landed_state", f"ch{THROW_CH}", "p_rads", "q_rads", "r_rads"])
        w.writerows(rows)
    with open(f"{base}.json", "w") as f:
        json.dump(dict(meta, results=results), f, indent=2)
    for ext in (".csv", "_steps.csv", ".json"):                 # on the SD card NOW: the battery may be pulled
        fd = os.open(f"{base}{ext}", os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)


def on_sigterm(*_):
    raise KeyboardInterrupt


def main(args):
    if args.mode is None:                                    # no --mode: mode.txt (autostart), else ppo-policy
        txt = Path(MODE_FILE).read_text().strip() if Path(MODE_FILE).exists() else ""
        args.mode = txt if txt in ("attitude-rule", "ppo-policy") else "ppo-policy"
    signal.signal(signal.SIGTERM, on_sigterm)               # systemctl stop = Ctrl+C (parameters put back, saved)
    net = None
    if args.mode == "ppo-policy" or (args.check and Path(ACTOR).exists()):
        if not Path(ACTOR).exists():
            raise SystemExit(f"No {ACTOR}: run on the PC  py .\\export_ppo_onnx.py  and copy the .onnx to the Pi")
        net = OnnxActor(ACTOR)
        print(f"PPO (ONNX) loaded: {net.source}")
    link = RealLink(args.conn, baud=args.baud, rate_hz=50)
    if args.check:                                           # --check works indoors too: attitude is enough
        t_end = time.time() + 10
        while time.time() < t_end and link.q is None:
            link.update()
            time.sleep(0.05)
        if link.q is None:
            raise SystemExit("No ATTITUDE_QUATERNION from the FC in 10 s - check SERIAL2 (2 / 921) and the wiring")
    else:
        for _ in range(8):                                   # EKF gives a position only after GPS fix + origin
            try:
                link.wait_state(timeout=15)
                break
            except RuntimeError:
                print("waiting for the EKF position (GPS fix) ...")
        else:
            raise SystemExit("No position from the EKF after 2 min - check GPS (CH6 middle at power-up)")
    for msg_id, hz in ((M.MAVLINK_MSG_ID_RAW_IMU, 50), (M.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 10),
                       (M.MAVLINK_MSG_ID_RC_CHANNELS, 20), (M.MAVLINK_MSG_ID_SYS_STATUS, 2),
                       (M.MAVLINK_MSG_ID_GPS_RAW_INT, 2), (M.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 2)):
        link.request_rate(msg_id, hz)
    t_end, t_ask = time.time() + 20, time.time()
    need = {"RAW_IMU", "EXTENDED_SYS_STATE", "RC_CHANNELS"} | set(DATA_MSGS)
    while time.time() < t_end and not need <= set(link.msgs):
        link.update()
        if time.time() - t_ask > 2.0:                        # change 2: ask again (a request may be lost)
            ask_data_again(link)
            t_ask = time.time()
        time.sleep(0.05)
    if need - set(link.msgs):
        raise SystemExit(f"The FC does not send {', '.join(sorted(need - set(link.msgs)))} - check SERIAL2 (2 / 921)")
    names = ["MOT_SPIN_ARM", "MOT_SPOOL_TIME", f"RC{THROW_CH}_OPTION", "GUID_OPTIONS", "MOT_THST_HOVER"]
    p = link.get_params(names)
    hover = float(p.get("MOT_THST_HOVER", 0.149))
    print(f"Connected | {args.mode} | hover thrust {hover:.3f} | MOT_SPIN_ARM {p.get('MOT_SPIN_ARM')} | "
          f"MOT_SPOOL_TIME {p.get('MOT_SPOOL_TIME')} | RC{THROW_CH}_OPTION {p.get(f'RC{THROW_CH}_OPTION')}")
    if abs(p.get("MOT_SPIN_ARM", 1.0)) < 1e-6 and p.get("MOT_SPOOL_TIME", 0.5) > 0.3:
        print("WARNING: MOT_SPIN_ARM 0 with MOT_SPOOL_TIME > 0.3 - motors start ~0.6 s after release (SITL 8 Nov: "
              "roll throws drifted 23 m). See guide C3: MOT_SPOOL_TIME 0.25")
    if args.check:
        try:
            check(link, p, net)
        except KeyboardInterrupt:
            print("\ncheck stopped")
        return
    guid, limits = restore_left_values(link, int(p.get("GUID_OPTIONS", 0)), ic.read_limits(link))
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / f"throw_real_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    meta = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "mode": args.mode, "hover": hover,
            "model": net.source if net else None, "throw_ch": THROW_CH, "hold_s": HOLD_S, "max_up_m": MAX_UP_M,
            "freefall_g": FREEFALL_G, "freefall_n": FREEFALL_N, "push_g": PUSH_G, "limits": limits,
            "version": "f (12 Nov 2026)", "spin_r_m": SPIN_R_M, "ground_guard_m": GROUND_GUARD_M,
            "guid_options": guid, "mot_spool_time": p.get("MOT_SPOOL_TIME")}
    results, rows, n = [], [], 0
    ch_was, t_print = rc(link, THROW_CH), 0.0
    beep(link, TUNE_START[args.mode])
    print(f"Pi READY TO USE ({args.mode}: {'1 long + 1 short' if args.mode == 'attitude-rule' else '1 long + 2 short'} beep)")
    print(f"WAIT - CH8 ALT_HOLD, stick LOW (keep it low), arm with CH9 (props still), then CH{THROW_CH} ON. Ctrl+C to end.")
    try:
        while True:
            link.update()
            ch = rc(link, THROW_CH)
            if ch > CH_HIGH >= ch_was:                           # throw switch just went ON
                p.update(link.get_params(["MOT_SPIN_ARM", "MOT_SPOOL_TIME", f"RC{THROW_CH}_OPTION"]))
                ok, out = checks(link, p)
                if not ok:
                    beep(link, TUNE_REFUSED)
                    print("REFUSED: " + "; ".join(f"{nm} ({txt})" for nm, good, txt in out if not good))
                else:
                    e = one_throw(link, args.mode, n + 1, rows, hover, guid, limits, net)
                    if e is not None:
                        n += 1
                        e["checks"] = {nm: txt for nm, _, txt in out}
                        results.append(e)
                        save(base, results, rows, meta)          # after EVERY throw
                        f = (lambda v: "-" if v is None else f"{v:.2f}")
                        print(f"Throw {n} | {args.mode} | motors on {f(e['motors_on_s'])} s | caught {f(e['caught_s'])} s"
                              f" | recovered {f(e['recovered_s'])} s | lowest {f(e['low_m'])} m (hand = 0) | "
                              f"peak rate {e['peak_rate_rads']:.1f} | drift {f(e['drift_m'])} m | {e['ended']}"
                              f" -> {e.get('hold_mode')} | braked {e.get('brake_m')} m, held within {e.get('wander_m')} m")
                        print(f"Next throw: CH{THROW_CH} OFF, arm again (CH9), CH{THROW_CH} ON.")
                back_to_alt_hold(link)                          # change 1: arm again without CH8
                ch = rc(link, THROW_CH)
            ch_was = ch
            if time.time() - t_print > 2.0:
                ask_data_again(link)                            # change 2
                fix, sats = gps(link)
                print(f"WAIT | CH{THROW_CH} {ch} CH6 {rc(link, 6)} CH3 {rc(link, 3)} | GPS fix {fix} {sats} sats | "
                      f"{volts(link):.2f} V | {'ARMED' if link.armed else 'disarmed'} {link.mode}"
                      + (f"  <- CH{THROW_CH} is ON: flip it OFF, then ON to start" if ch > CH_HIGH else ""))
                t_print = time.time()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        ic.send_params(link, {"GUID_OPTIONS": guid})
        if limits:
            ic.send_params(link, limits)
        save(base, results, rows, meta)
        print(f"Saved: {base}.csv / _steps.csv / .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None, choices=["attitude-rule", "ppo-policy"],
                    help="default: the word in mode.txt (autostart), else ppo-policy")
    ap.add_argument("--check", action="store_true", help="ladder step 0: link OK, switch OK - never arms")
    ap.add_argument("--conn", default="/dev/serial0", help="Pi UART (SITL test: tcp:127.0.0.1:5762)")
    ap.add_argument("--baud", type=int, default=921600)
    try:
        main(ap.parse_args())
    except KeyboardInterrupt:                                # change 3: Ctrl+C / systemctl stop at start-up
        print("stopped (before the WAIT loop - no throws, nothing to save)")
