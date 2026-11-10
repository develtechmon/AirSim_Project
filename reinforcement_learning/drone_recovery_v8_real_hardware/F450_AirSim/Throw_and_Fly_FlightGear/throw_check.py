r"""
throw_check.py - THROW TEST: throw the drone into the air, then catch it (ArduPilot SITL, built-in physics)
==========================================================================================================

1. HOW TO RUN  (Windows PowerShell, in the folder with these scripts, e.g. Throw_and_Fly\ - SITL must run in WSL, section 3)
------------------------------------------------------------------------------------------------------
    ArduPilot Throw mode vs the attitude rule, straight-up and rolling throws (~3 min, no torch needed):

    py .\throw_check.py

    + your Stage 3 PPO (adds mode "ppo-policy"):

    py .\throw_check.py --model .\models\round1_final.zip

    More throws per case / other throws:

    py .\throw_check.py --repeats 3 --types up roll pitch

    Old ending (GUIDED hold 3 s, fly back to the throw spot) instead of LOITER where it recovered:

    py .\throw_check.py --fly-back

    GAZEBO F450 (the drone throws itself - section 9):

    py .\throw_check.py --launch --types up roll pitch

    py .\throw_check.py --launch --modes attitude-rule ppo-policy --model .\models\round1_final.zip --repeats 3 --types up roll pitch


2. WHAT IT DOES  (one throw = one line in the results)
------------------------------------------------------------------------------------------------------
    | Step | ArduPilot "throw-mode"                    | Script modes (attitude-rule, ppo-policy, bc-policy)  |
    |------|-------------------------------------------|------------------------------------------------------|
    | 1    | mode THROW, arm (motors stopped)          | mode GUIDED_NOGPS, arm (motors idle on the ground)   |
    | 2    | the script THROWS the drone: SIM_SHOVE (push) + SIM_TWIST (spin) for 0.3 s                       |
    | 3    | ArduPilot detects the throw itself        | the script detects it: accelerometer < 0.35 g for    |
    |      | (needs GPS - see section 4)               | 3 samples in a row (free fall = released)            |
    | 4    | ArduPilot spools up, uprights, holds      | motors spool up (max 1 s), then the recovery flies:  |
    |      |                                           | level + thrust (rule) or the network (PPO / BC)      |
    | 5    | when calm: ArduPilot LOITER holds the recovery point 10 s (the script does the RECOVERY only), |
    |      | measured: braking distance + how still it holds. Then LAND right there, check it disarmed.     |
    |      | (--fly-back: GUIDED hold 3 s, fly back to the throw spot - section 8)                          |
    "Released" (the moment the hand lets go) is found the SAME way for every mode: first accelerometer
    reading below 0.35 g after the push. All times in the results count from that moment.

    | Throw  | Push (body, m/s^2, 0.3 s) | Spin (rad/s^2, 0.3 s) | Roughly                                  |
    |--------|---------------------------|-----------------------|------------------------------------------|
    | up     | 30 up                     | none                  | ~6 m/s up, top ~2.5 m                    |
    | roll   | 45 up                     | 40 roll               | ~12 rad/s roll, tumbles ~1 turn          |
    | pitch  | 45 up                     | 40 pitch              | same about pitch                         |


3. SIMULATOR  (WSL - ArduPilot's own physics, NOT Gazebo: SIM_SHOVE / SIM_TWIST only exist there)
------------------------------------------------------------------------------------------------------
    Simulator (terminal 2) - the setup used for ALL numbers in section 4:

    cd ~/sitl_throw

    sim_vehicle.py -v ArduCopter -f quad --console --map

    Optional 3D view (FlightGear, bright daylight, scenery included) - BOTH programs at KSFO:
      terminal 1:   LIBGL_ALWAYS_SOFTWARE=1 ~/ardupilot/Tools/autotest/fg_quad_view.sh --airport=KSFO
      terminal 2:   sim_vehicle.py -v ArduCopter -f quad --console --custom-location=37.6188,-122.3754,5,0
      (5 = ground height: with 4 the drone sat 0.5 m inside FlightGear's ground. FlightGear sets "noon" for
       its OWN start airport: a different place in the two commands = night / sea under the drone.)

    TIMING CHECK - every run: "motors on" of the script modes must be ~0.3 s.
    | motors on | Meaning                                                                               |
    |-----------|---------------------------------------------------------------------------------------|
    | ~0.3 s    | simulator keeps real time -> numbers valid                                            |
    | ~1.0 s    | simulator SLOWER than real time -> numbers NOT valid (catch too late, crash at 6.6 m/s).|
    |           | Reproduced 25 Oct with --speedup 0.5. Seen with FlightGear (1400x900, software drawing)|
    |           | and at KSFO with --map. Close FlightGear / drop --map, restart sim_vehicle, run again.|

    Position source = whatever ~/sitl_throw has saved. The script prints which one at the start.
      optical flow : EK3_SRC1_POSXY 0, EK3_SRC1_VELXY 5, EK3_SRC1_POSZ 1, EK3_SRC1_VELZ 0
                     (POSZ 1 = barometer height: the tested setting; with 2 = rangefinder the SITL
                      height jumped while the simulated rangefinder read 0 on the ground)
                     (+ FLOW_TYPE 10, SIM_FLOW_ENABLE 1, RNGFND1_TYPE 100)
      GPS          : EK3_SRC1_POSXY 3, EK3_SRC1_VELXY 3, EK3_SRC1_POSZ 1, EK3_SRC1_VELZ 3
    Change with "param set ..." in the MAVProxy console, then "reboot".


4. EXPECTED RESULT  (measured 25 Oct 2026, ArduCopter 4.4 SITL, quad, 1 run per line)
------------------------------------------------------------------------------------------------------
    Optical flow (EK3_SRC1_POSXY 0, VELXY 5, POSZ 1 = barometer height), 2 repeats each:
    | Throw | throw-mode (ArduPilot)                                  | attitude-rule (script)                  |
    |-------|---------------------------------------------------------|-----------------------------------------|
    | up    | 0/2: 1x "could not arm" (Need Position Estimate),        | 2/2, motors on 0.26-0.28 s, calm 0.52 s, |
    |       |      1x armed but NOT detected -> hit the ground        | drift 0.0 m, top 2.8 m                   |
    | roll  | 0/2: not detected -> hit the ground (up to 8.5 m/s)      | 2/2, on 0.34-0.38 s, calm 1.24 s, 0.9 m  |
    | pitch | 0/2: not detected -> hit the ground                      | 2/2, on 0.24-0.34 s, calm 1.22 s, 0.7-1 m|
    Why: mode_throw.cpp only detects with an ABSOLUTE position (horiz_pos_abs); EKF3 sets that only for
    GPS (AP_NavEKF3_Outputs.cpp line 726). Optical flow = relative position. So ArduPilot's Throw mode
    can never catch with optical flow - a RESULT to record, not a bug.

    GPS (default SITL settings), 1 run each:
    | Throw | throw-mode                                   | attitude-rule                               |
    |-------|----------------------------------------------|---------------------------------------------|
    | up    | caught: motors on 0.72 s, calm 2.04 s, fell  | caught: motors on 0.32 s, calm 0.50 s       |
    |       | from 4.4 to 2.0 m, drift 0.1 m               | drift 0.0 m                                 |
    | roll  | caught: motors on 0.68 s, calm 2.38 s,       | caught: motors on 0.26 s, calm 1.26 s,      |
    |       | drift 13.2 m                                 | drift 0.4 m                                 |
    The script reacts faster because it detects the free fall from the accelerometer at once; Throw
    mode waits until the speed has dropped 2.5 m/s (its confirmation rule).

    IMPORTANT - SITL fix needed for rolling throws with the simulated rangefinder (section 6).


5. OUTPUT FILES  (folder runs\throw_check\)
------------------------------------------------------------------------------------------------------
    throw_check_<date>.csv         one line per throw (the table below)
    throw_check_<date>_steps.csv   every 50 Hz step: phase, tilt, rate, height, speed, accel, p/q/r
    throw_check_<date>.json        settings (throws, modes, position source, hover) + summary

    | Column          | Meaning                                                                    |
    |-----------------|----------------------------------------------------------------------------|
    | detected        | the catch started (motors spooled up) before hitting the ground            |
    | motors_on_s     | release -> motors at full range (detection + spool-up)                     |
    | upright_s       | release -> tilt back below 30 deg (0 if it never tilted past 30)           |
    | caught_s        | release -> hand-back: spin < 1.2 rad/s, tilt < 30 deg, falling < 1 m/s;   |
    |                 | script modes then hold the height themselves (level + vertical-speed thrust)|
    | recovered_s     | release -> calm for 0.5 s (rate < 1.2 rad/s, tilt < 30 deg, h-speed < 3,   |
    |                 | vertical speed < 1 m/s, height > 0.5 m)                                    |
    | apex_m / low_m  | highest / lowest point after the top, metres ABOVE THE GROUND (the EKF     |
    |                 | height measured while waiting on the ground = 0)                           |
    | drift_m         | horizontal distance from the throw spot after the 3 s hold                 |
    | crashed         | hit the ground ("SIM Hit ground" or height < 0.15 m while falling)         |
    | hold_mode       | LOITER (or "script height hold" if LOITER refused: no position yet)        |
    | brake_m/brake_s | after the hand-over to LOITER: distance / time until slower than 0.2 m/s   |
    | wander_m        | then: max distance from that stopped point = HOW STILL IT HOVERS           |
    | hold_dz_m       | max height change during the 10 s hold                                     |


6. SITL FIX FOR ROLLING THROWS  (once, WSL - simulator only, does not touch any real firmware)
------------------------------------------------------------------------------------------------------
    ArduPilot 4.4 SITL returns an infinite rangefinder distance when the drone tilts past 90 deg
    (libraries/SITL/SIM_Aircraft.cpp, rangefinder_range()). Later code turns it into an integer and SITL
    stops with "ERROR: Floating point exception - aborting" (seen in the first roll throw, 25 Oct).
    Fix = report 99 m ("nothing in view", out of range) instead, then rebuild SITL:

    sed -i 's/        return INFINITY;/        return 99.0f;/' ~/ardupilot/libraries/SITL/SIM_Aircraft.cpp

    cd ~/ardupilot

    ./waf copter

    Check: grep -n "return 99.0f" ~/ardupilot/libraries/SITL/SIM_Aircraft.cpp  -> one line.


7. SAFETY NOTES FOR THE REAL DRONE LATER
------------------------------------------------------------------------------------------------------
    Script modes arm in GUIDED_NOGPS: on a real drone the props would spin at idle IN YOUR HAND.
    Real throws need MOT_SPIN_ARM 0 (armed but props still) - not used in this SITL test.
    Real IMUs are not at the centre of the drone: a spin adds centripetal acceleration, so the
    0.35 g free-fall limit must be checked on real logs before any real throw.
    Optical flow keeps a position only while the rangefinder sees the ground (section 8): stay well below
    its range on the real drone too.


8. AFTER THE RECOVERY, LANDING AND GUARDS  (updated 25 Oct 2026)
------------------------------------------------------------------------------------------------------
    DEFAULT = option B "recovery only": the script catches, then ArduPilot LOITER holds the point where it
    recovered (same as Throw mode's docs: switch to Loiter after the throw), 10 s, then LAND right there.
    SITL: the throttle stick rests at 1000 = LOITER would descend -> the script holds it at 1500 (RC
    override, throttle only). Real drone: YOUR thumb keeps the throttle stick centred.
    Measured 25 Oct (attitude-rule, optical flow, SITL):
    | Throw | recovered | LOITER braked     | then held within | height +/- |
    |-------|-----------|-------------------|------------------|------------|
    | up    | 0.46-0.52 s | 0.0 m             | 0.08-0.09 m      | 0.1-0.3 m  |
    | roll  | 1.16-1.18 s | 0.59 m in 1.14 s  | 0.28 m           | 0.2-0.45 m |
    | pitch | 1.12-1.16 s | 0.59 m in 1.14 s  | 0.36 m           | 0.2-0.4 m  |
    The recovery hands over at up to ~0.9 m/s (calm = < 3 m/s), so LOITER first brakes ~0.6 m.
    Real drone: wind, flow texture and the tune decide the real numbers - measure them the same way.

    --fly-back (old ending) and the landing guards:
    | Situation                                       | What the script does                              | Printed                          |
    |-------------------------------------------------|---------------------------------------------------|----------------------------------|
    | after every throw                               | take-off point = THIS throw's spot (was: first    | [link] take-off point saved      |
    |                                                 | throw only -> one fly-away made every later throw |                                  |
    |                                                 | try to fly 28 m back for 90 s)                     |                                  |
    | optical flow, higher than 3 m                   | straight down to 3 m above the spot, then fly back| [link] descending from 6.7 m ... |
    | height checks                                   | measured from the throw spot, not the raw EKF     |                                  |
    |                                                 | height (SITL flow EKF read ~4 m too low)          |                                  |
    | still armed after landing (30 s), before the    | LAND, wait until not sinking for 2 s (= on the    | [sim] drone still armed (...) -  |
    | next throw, or at the start of the script       | ground), forced disarm, then carry on (sim only)  | LAND, then forced disarm         |
    | GUID_OPTIONS thrust bit / ATC_ACCEL_* = 0 left  | puts ArduPilot's values back (thrust bit off,     | [sim] test values left from a    |
    | (script stopped in the middle of a throw)       | 110000 / 110000 / 27000), then carries on         | stopped run - restored           |
    Why 3 m: SITL rangefinder max = RNGFND1_MAX_CM 700 (7 m, ArduPilot default). Climb test 25 Oct: above
    ~7.5 m the EKF dropped "relative position" (EKF_STATUS flags 0x16f -> 0x027) - GUIDED then has nothing
    to steer by. Your run 25 Oct: flying back at 6.4 m drifted to 29.6 m. My SITL flew back fine at 6.6 m
    (3/3), so the height is a safety margin; the exact trigger on your PC is not proven yet.
    Measured with the new landing (25 Oct, flow, 2 each): up / roll / pitch recovered 0.53 / 1.25 / 1.23 s,
    all landed 0.03-0.09 m from the throw spot and disarmed by themselves.
    If it still cannot disarm: "STOP ... Restart sim_vehicle" (Ctrl+C, same command).


9. GAZEBO F450: --launch  (added 25 Oct 2026)
------------------------------------------------------------------------------------------------------
    Gazebo does the physics itself, so the throw push (SIM_SHOVE / SIM_TWIST, ArduPilot's own physics only:
    SIM_Aircraft.cpp / SIM_Multicopter.cpp) does NOTHING there. With --launch the drone throws itself:

    | Step | What the script does                                              | Like a hand throw?               |
    |------|-------------------------------------------------------------------|----------------------------------|
    | 1    | climb to LAUNCH_ALT 3 m (overshoots to ~4.5 m), hold 2 s           | = the hand height                |
    |      | and MEASURE the hover thrust (VFR_HUD throttle)                   |                                  |
    | 2    | push: full thrust, level, 0.6 s                                   | = the arm pushing up             |
    | 3    | spin kick (roll/pitch): lead target + 9 rad/s until 80 % reached   | = the wrist flick                |
    | 4    | release: thrust 0 (motors at minimum), keep tumbling              | = let go: free fall < 0.35 g      |
    | 5    | 0.3 s more at minimum (LAUNCH_FREE_S), then the catch starts       | = the 0.3 s spool-up of a real   |
    |      | ("motors on" = this delay + detection, ~0.34 s)                   |   throw (motors were idling)     |
    | 6    | same catch -> LOITER 10 s -> LAND as section 8                     |                                  |
    Only attitude-rule / ppo-policy / bc-policy (Throw mode needs a real push with stopped motors).

    Run (WSL): Gazebo F450 world, then SITL from its own folder (Gazebo F450 guide, Step 2):

    gazebo --verbose ~/ardupilot_gazebo/worlds/f450_airsim_runway.world

    cd ~/sitl_f450

    sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map

    Then (PowerShell):  py .\throw_check.py --launch --types up roll pitch

    Why the hover thrust is measured: with MOT_THST_HOVER at its default (0.39) the F450 test drone only
    balanced its weight in the catch and sank into the ground at 2 m/s (3/3 crashed). Measured 0.58-0.59
    -> 6/6 caught. The line "launch: hovering at ... hover thrust measured ..." shows the value used.

    Why 9 rad/s: the spin decides how tilted the drone is when the catch starts (F450 test, 25 Oct):
    | Spin kick | Tilt at catch start          | Recovered  | Lowest   |
    |-----------|------------------------------|------------|----------|
    | 12 rad/s  | 4 deg (lucky: back to level)  | 0.84 s     | 3.0 m    |
    | 9 rad/s   | 85-111 deg (on its side)      | 1.0-1.26 s | 1.7-2.6 m| <- used (fair and hard)
    | 7 rad/s   | 141-154 deg (upside down)     | 1.28 s / CRASHED (lowest 0.2 m from 2 m) |
    All roll/pitch launches tumble ~1 turn through upside down (max tilt 174-179 deg), peak 10.5-11 rad/s.

    Expected (attitude-rule, 3 m launch, 9 rad/s, 2 repeats, 25 Oct) - measured on ArduPilot's own physics
    with an F450 frame file (1.0 kg, 455 mm, inertia 0.0067/0.0080/0.0143, T/W 1.63, MOT_THST_EXPO 0.65),
    NOT in Gazebo (no Gazebo on the test PC). Your Gazebo numbers are the real result:
    | Throw | recovered   | lowest    | drift     | LOITER: braked / held within |
    |-------|-------------|-----------|-----------|------------------------------|
    | up    | 0.70-0.72 s | 4.1 m     | 0.6 m     | 0 m / 0.04-0.05 m            |
    | roll  | 1.02-1.06 s | 3.2-3.3 m | 1.2-1.4 m | 0.09-0.16 m / 0.1 m          |
    | pitch | 1.04-1.06 s | 3.1-3.3 m | 1.1-1.2 m | 0.18-0.22 m / 0.1 m          |
    PPO with --launch: not tested yet (no model on the test PC) - your first run.
    Not comparable with section 4: different drone (F450, T/W 1.63), launch from ~4.5 m, motors already idling.
"""

import argparse
import csv
import datetime
import json
import math
import time
from pathlib import Path

import numpy as np

import impact_check as ic
from ap_link import APLink, RateKeeper, M

HZ = 50.0
FREEFALL_G = 0.35          # accelerometer below this = free fall (released)
FREEFALL_N = 3             # ... for this many samples in a row (60 ms)
IDLE_S = 1.5               # s on the ground, armed, before the throw
WAIT_MAX_S = 4.0           # s after the push: no release / no catch by then = not detected
SPOOL_MAX_S = 1.0          # s max for the motors to reach full range after the script detects the throw
RECOVER_MAX_S = 10.0       # s after release: not calm by then = not recovered
HOLD_S = 3.0               # s hold after the catch (drift measurement)
GROUND_M = 0.15            # m: below this while falling = hit the ground
ALT_OK = 0.5               # m: "recovered" also needs this height (throws start on the ground)
VZ_OK = 1.0                # m/s: ... and no longer rising or falling faster than this (caught, not still falling)
HOLD_K = 0.4               # x hover per m/s: height hold after the catch (script modes)
LANDED_ON_GROUND = 1       # EXTENDED_SYS_STATE.landed_state: 1 on ground, 2 in air
HOME_FLY_ALT = 3.0         # m above the throw spot: optical flow flies back at most this high (rangefinder
                           # max 7 m - above it the EKF loses the position and the drone drifts away)
LOITER_S = 10.0            # s LOITER hold at the recovery point (default after a recovery, option B)
STOP_V = 0.2               # m/s: LOITER has finished braking (the hold accuracy is measured from here)
STICK_MID = 1500           # throttle stick centre: LOITER holds height (SITL stick rests at 1000 = descend)
STOCK_LIMITS = {"ATC_ACCEL_R_MAX": 110000.0, "ATC_ACCEL_P_MAX": 110000.0,   # ArduPilot 4.4 defaults
                "ATC_ACCEL_Y_MAX": 27000.0}                                 # (AC_AttitudeControl.h)
GROUND_STILL_S = 2.0       # s in LAND without sinking (|vz| < 0.2 m/s) = on the ground -> forced disarm
LAND_WAIT_S = 30.0        # s for LAND to disarm by itself (from 3 m: ~6 s down + ~1.5 s land detector)
OUT = Path("runs") / "throw_check"

THROWS = {                 # push (body m/s^2, z negative = up), spin (body rad/s^2), duration (ms)
    "up":    {"shove": (0.0, 0.0, -30.0), "twist": (0.0, 0.0, 0.0), "ms": 300},
    "roll":  {"shove": (0.0, 0.0, -45.0), "twist": (40.0, 0.0, 0.0), "ms": 300},
    "pitch": {"shove": (0.0, 0.0, -45.0), "twist": (0.0, 40.0, 0.0), "ms": 300},
}
LAUNCH = {                 # --launch (Gazebo): the drone throws ITSELF - body rates of the spin kick (rad/s)
    "up":    (0.0, 0.0, 0.0),
    "roll":  (9.0, 0.0, 0.0),
    "pitch": (0.0, 9.0, 0.0),
}
LAUNCH_ALT = 3.0           # m: climb to this "hand height" first (Gazebo has no SIM_SHOVE push); the weak F450
                           # (T/W 1.63) caught on its side lost up to 2.1 m (25 Oct), so 3 m (it overshoots ~1 m)
LAUNCH_PUSH_S = 0.6        # s full thrust, level = the push up
LAUNCH_SPIN_S = 0.35       # s max spin kick (stops earlier at 80 % of the spin rate)
LAUNCH_FREE_S = 0.30       # s motors at minimum after the release = the same delay as a hand throw's spool-up
SCRIPT_MODES = ("attitude-rule", "bc-policy", "ppo-policy")
MODES = ("throw-mode",) + SCRIPT_MODES
SRC_NAMES = {0: "none", 3: "GPS", 5: "optical flow", 6: "external nav"}   # EK3_SRC1_* values


def accel_g(link):
    """Accelerometer magnitude in g from RAW_IMU (ArduPilot sends milli-g), None if not received yet."""
    m = link.msgs.get("RAW_IMU")
    if m is None:
        return None
    return math.sqrt(m.xacc ** 2 + m.yacc ** 2 + m.zacc ** 2) / 1000.0


def landed_state(link):
    m = link.msgs.get("EXTENDED_SYS_STATE")
    return int(m.landed_state) if m is not None else 0


def wait_origin(link, timeout=60):
    """Optical flow has no GPS: ArduPilot needs an EKF origin before position modes work. Without it,
    entering GUIDED raised 'Internal Errors 0x100000' in the first SITL tests (25 Oct 2026)."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        link.set_ekf_origin()
        link.m.mav.command_long_send(link.m.target_system, link.m.target_component, M.MAV_CMD_REQUEST_MESSAGE,
                                     0, M.MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN, 0, 0, 0, 0, 0, 0)
        t1 = time.time() + 1.0
        while time.time() < t1:
            link.update()
            if "GPS_GLOBAL_ORIGIN" in link.msgs:
                return True
            time.sleep(0.05)
    return False


def throw(link, spec):
    """The push + spin. TIME parameters last: setting them starts the effect (SITL.cpp / SIM_Aircraft.cpp)."""
    sx, sy, sz = spec["shove"]
    tx, ty, tz = spec["twist"]
    ic.send_params(link, {"SIM_SHOVE_X": sx, "SIM_SHOVE_Y": sy, "SIM_SHOVE_Z": sz,
                          "SIM_TWIST_X": tx, "SIM_TWIST_Y": ty, "SIM_TWIST_Z": tz})
    time.sleep(0.2)
    ic.send_params(link, {"SIM_SHOVE_TIME": spec["ms"], "SIM_TWIST_TIME": spec["ms"] if any(spec["twist"]) else 0})


def send_target(link, q, rates, thrust):
    """SET_ATTITUDE_TARGET with attitude + body rates + thrust (type_mask 0), as impact_check's lead hit."""
    with link.io_lock:
        link.m.mav.set_attitude_target_send(0, link.m.target_system, link.m.target_component, 0,
                                            [float(v) for v in q], float(rates[0]), float(rates[1]),
                                            float(rates[2]), float(thrust))


def launch_takeoff(link, hgt, rk, rows, n, kind, mode, spot, hover):
    """--launch: climb (climb-rate mode, the normal take-off path) to LAUNCH_ALT, settle, and MEASURE the hover
    thrust (VFR_HUD throttle while ArduPilot holds the height). Returns the hover thrust for the catch.
    Why: with MOT_THST_HOVER still at its default (0.39) the F450-like test drone only balanced its weight in
    the catch and sank into the ground at 2 m/s (25 Oct)."""
    t0 = time.time()
    while hgt() < LAUNCH_ALT and time.time() - t0 < 15:
        ic.send_attitude(link, level_q(link), 0.8)           # climb-rate mode: 0.8 = climb
        rk.wait()
        link.update()
        log_row(rows, n, kind, mode, "takeoff", time.time() - t0, link, spot, accel_g(link))
    t1, thr = time.time(), []
    while time.time() - t1 < 2.0:
        ic.send_attitude(link, level_q(link), 0.5)           # hold
        rk.wait()
        link.update()
        hud = link.msgs.get("VFR_HUD")
        if hud is not None and time.time() - t1 > 1.0 and abs(float(link.vel[2])) < 0.3:
            thr.append(hud.throttle / 100.0)                 # ArduPilot's throttle while holding height
    if hgt() < 0.8 * LAUNCH_ALT:
        raise SystemExit(f"--launch: could not climb to {LAUNCH_ALT} m (now {hgt():.1f} m)")
    meas = float(np.median(thr)) if len(thr) >= 10 else None
    used = meas if meas is not None and 0.1 < meas < 0.95 else hover
    print(f"   launch: hovering at {hgt():.1f} m | hover thrust measured "
          f"{'-' if meas is None else f'{meas:.2f}'} (MOT_THST_HOVER {hover:.2f}) -> using {used:.2f}")
    return used


def launch_cmd(link, t_since, spin, hover, released):
    """--launch commands (thrust mode): push up -> spin kick -> release (motors at minimum, keep tumbling)."""
    if not released and t_since < LAUNCH_PUSH_S:
        return level_q(link), (0.0, 0.0, 0.0), 1.0                       # push: full thrust, level
    rate_now = float(np.linalg.norm(link.rates))
    kick = any(spin) and t_since < LAUNCH_PUSH_S + LAUNCH_SPIN_S and rate_now < 0.8 * float(np.linalg.norm(spin))
    if not released and kick:
        return ic.lead_quat(link.q, spin), spin, hover                   # spin kick (lead target + rate)
    if any(spin):
        return ic.lead_quat(link.q, spin), spin, 0.0                     # released: tumble on, motors minimum
    return level_q(link), (0.0, 0.0, 0.0), 0.0                           # released, no spin


def level_q(link):
    """Level attitude, current heading."""
    R = ic.quat_to_R(link.q)
    return ic.euler_quat(0.0, 0.0, math.atan2(R[1, 0], R[0, 0]))


def height_hold(link, hover):
    """After the catch: level + thrust that brings the vertical speed to 0 (thrust mode stays on).
    Switching GUIDED_NOGPS back to climb-rate mode in the air raised 'Internal Errors 0x100000'
    (AC_PosControl::update_z_controller ran without init) in the SITL tests of 25 Oct 2026."""
    up = max(0.5, 1.0 - 2.0 * (link.q[1] ** 2 + link.q[2] ** 2))   # cos(tilt)
    thr = hover * (1.0 + HOLD_K * float(link.vel[2])) / up             # vel[2] > 0 = falling -> more thrust
    return level_q(link), float(np.clip(thr, 0.3 * hover, 1.0))


def wait_disarmed(link, timeout=60):
    t_end = time.time() + timeout
    while time.time() < t_end:
        link.update()
        if not link.armed:
            return True
        time.sleep(0.1)
    return False


def force_disarm(link):
    link.m.mav.command_long_send(link.m.target_system, link.m.target_component,
                                 M.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 21196, 0, 0, 0, 0, 0)
    t_end = time.time() + 3
    while time.time() < t_end:
        link.update()
        time.sleep(0.05)
    return not link.armed


def make_disarmed(link, why):
    """Simulator only: if the drone is still armed, put it in LAND, wait until it is really on the ground
    (not sinking for GROUND_STILL_S), then force-disarm. Works even when ArduPilot's land detector never
    confirms the landing (seen 25 Oct). Returns True when disarmed."""
    link.update()
    if not link.armed:
        return True
    print(f"[sim] drone still armed ({why}) - LAND, then forced disarm on the ground (simulator only)")
    rc_throttle(link, 0)                                     # no stick override left over
    t_end, t_req, still_since = time.time() + 2 * LAND_WAIT_S, 0.0, None
    while time.time() < t_end and link.armed:
        link.update()
        if link.mode != "LAND" and time.time() - t_req > 1.0:
            link.m.set_mode("LAND")
            t_req = time.time()
        sinking = link.vel is not None and abs(float(link.vel[2])) > 0.2
        if link.mode == "LAND" and not sinking:
            still_since = still_since or time.time()
            if time.time() - still_since >= GROUND_STILL_S and force_disarm(link):
                break
        else:
            still_since = None
        time.sleep(0.05)
    link.update()
    print("[sim] disarmed" if not link.armed else "[sim] WARNING: still armed - restart sim_vehicle")
    return not link.armed


def land_and_disarm(link, flow, go_home=True):
    """Fly back to the throw spot (--fly-back; optical flow: at most HOME_FLY_ALT high) or land in place,
    and make sure it is DISARMED - a drone left armed makes the next throw start wrong (seen 25 Oct:
    next run started already armed, 31 m away). Simulator only (needs SIM_SHOVE_TIME)."""
    if link.land(timeout=LAND_WAIT_S, go_home=go_home, fly_alt=HOME_FLY_ALT if flow else None):
        return True
    return make_disarmed(link, "LAND did not disarm it")


def hit_ground(link, t_since):
    """SITL prints 'SIM Hit ground at X m/s' when the drone lands hard. Returns the speed or None."""
    for t, text in link.texts:
        if t >= t_since and "Hit ground" in text:
            try:
                return float(text.split("at")[1].split("m/s")[0])
            except (IndexError, ValueError):
                return 0.0
    return None


def log_row(rows, n, kind, mode, phase, t, link, spot, g):
    s = link
    hs = float(np.linalg.norm(s.vel[0:2]))
    rows.append([n, kind, mode, phase, round(t, 3), round(s.tilt_deg, 1), round(float(np.linalg.norm(s.rates)), 2),
                 round(s.alt, 2), round(hs, 2), round(float(np.linalg.norm(s.pos[0:2] - spot[0:2])), 2),
                 round(g, 3) if g is not None else "", landed_state(s)] + [round(float(v), 2) for v in s.rates])


def rc_throttle(link, pwm):
    """RC override of the THROTTLE only (other sticks untouched). pwm 0 = give it back to the RC."""
    with link.io_lock:
        link.m.mav.rc_channels_override_send(link.m.target_system, link.m.target_component,
                                             0, 0, int(pwm), 0, 0, 0, 0, 0)


def loiter_hold(link, n, kind, mode, rows, t_trig, spot, hover, rk):
    """Option B (default): at the recovery point hand over to ArduPilot LOITER, hold LOITER_S, measure how
    still it stays, then LAND right there. The script only does the RECOVERY.
    SITL: the throttle stick rests at 1000 (= LOITER descends), so the script holds it at 1500 (centre) -
    on the real drone that is your thumb on the throttle stick."""
    t_end, t_req = time.time() + 3.0, 0.0
    while time.time() < t_end and link.mode != "LOITER":
        rc_throttle(link, STICK_MID)                     # centre stick BEFORE LOITER starts
        if time.time() - t_req > 0.5:
            link.m.set_mode("LOITER")
            t_req = time.time()
        ic.send_attitude(link, *height_hold(link, hover))   # keep flying until LOITER takes over
        rk.wait()
        link.update()
    hold_mode = "LOITER" if link.mode == "LOITER" else "script height hold"
    if hold_mode == "LOITER":
        print("[link] mode LOITER - ArduPilot holds the recovery point")
    else:
        print("   LOITER not accepted (no position estimate yet) - script holds the height only")
    point, alt0 = link.pos.copy(), link.alt
    max_d = max_dz = 0.0
    stop_pt, t_stop, wander = None, None, 0.0           # point where it stopped braking, then how much it wanders
    note = None
    t0 = time.time()
    while time.time() - t0 < LOITER_S:
        if hold_mode == "LOITER":
            if link.mode != "LOITER":                    # pilot / failsafe changed the mode: hands off
                note = f"mode changed to {link.mode} during the hold"
                print(f"   {note} - the script stops holding")
                break
            rc_throttle(link, STICK_MID)
        else:
            ic.send_attitude(link, *height_hold(link, hover))
        rk.wait()
        link.update()
        max_d = max(max_d, float(np.linalg.norm(link.pos[0:2] - point[0:2])))
        if stop_pt is None and float(np.linalg.norm(link.vel[0:2])) < STOP_V:
            stop_pt, t_stop = link.pos.copy(), time.time() - t0
        if stop_pt is not None:
            wander = max(wander, float(np.linalg.norm(link.pos[0:2] - stop_pt[0:2])))
        max_dz = max(max_dz, abs(link.alt - alt0))
        log_row(rows, n, kind, mode, "loiter", time.time() - t_trig, link, spot, accel_g(link))
    info = {"hold_mode": hold_mode, "hold_s": round(time.time() - t0, 1), "hold_max_m": round(max_d, 2),
            "hold_end_m": round(float(np.linalg.norm(link.pos[0:2] - point[0:2])), 2),
            "hold_dz_m": round(max_dz, 2),
            "brake_s": round(t_stop, 2) if t_stop is not None else None,
            "brake_m": round(float(np.linalg.norm(stop_pt[0:2] - point[0:2])), 2) if stop_pt is not None else None,
            "wander_m": round(wander, 2) if stop_pt is not None else None, "drift": float(np.linalg.norm(link.pos[0:2] - spot[0:2]))}
    if note:
        info["note"] = note
    if link.mode == "LOITER":                            # land right here (no flying back)
        t_end = time.time() + 3
        while time.time() < t_end and link.mode != "LAND":
            rc_throttle(link, STICK_MID)
            link.m.set_mode("LAND")
            time.sleep(0.2)
            link.update()
    rc_throttle(link, 0)                                 # give the throttle back to the RC
    return info


def one_throw(link, kind, mode, n, rows, hover, guid, limits, after="loiter", launch=False, ff=None):
    spec = THROWS[kind]
    if not wait_disarmed(link, timeout=5) and not make_disarmed(link, "before the next throw"):
        raise SystemExit("STOP - still armed and could not be disarmed. Restart sim_vehicle (Ctrl+C, same command)")
    link.set_mode("THROW" if mode == "throw-mode" else "GUIDED_NOGPS")
    try:
        link.arm(retries=4 if mode == "throw-mode" else 10)
    except RuntimeError:
        if mode != "throw-mode":
            raise
        print("   throw-mode: ArduPilot refuses to arm (\"Need Position Estimate\": Throw mode needs a position "
              "on the ground - optical flow gives none until airborne)")
        return {"n": n, "type": kind, "mode": mode, "released": False, "detected": False, "release_s": None,
                "motors_on_s": None, "upright_s": None, "recovered": False, "recovered_s": None, "apex_m": None,
                "low_m": None, "peak_rate_rads": None, "drift_m": None, "crashed": False, "impact_speed_ms": None,
                "note": "could not arm: Need Position Estimate"}, False
    link.update()
    link.save_home()                                         # take-off point = THIS throw's spot
    spot = link.pos.copy()
    rk = RateKeeper(HZ)
    script = mode in SCRIPT_MODES
    t_end = time.time() + IDLE_S
    ground = []
    while time.time() < t_end:                               # idle on the ground, armed
        if script:
            ic.send_attitude(link, level_q(link), 0.5)       # climb-rate mode: 0.5 = no climb -> stays landed
        rk.wait()
        link.update()
        ground.append(link.alt)
    h0 = float(np.mean(ground[-10:]))                        # EKF height of the ground: heights below are ABOVE GROUND
    hgt = (lambda: link.alt - h0)
    if launch:                                               # Gazebo: no SIM_SHOVE - the drone throws itself
        hover = launch_takeoff(link, hgt, rk, rows, n, kind, mode, spot, hover)
        if limits:
            ic.send_params(link, {k: 0 for k in limits})     # rotation limits off (as the hit and the rules)
        if ff is not None:
            ic.send_params(link, {ic.FF_NAME: 0})            # lead target used directly (impact_check, 23 Oct)
        ic.send_params(link, {"GUID_OPTIONS": int(guid) | ic.GUID_THRUST_BIT})   # thrust = real thrust
        time.sleep(0.1)
    t_trig = time.time()
    if not launch:
        throw(link, spec)
    phase, n_ff = "push", 0
    t_free_end = None
    t_rel = t_motors = t_upright = t_rec = held = t_caught = None
    apex, low = hgt(), None
    tilted = False
    peak_rate = 0.0
    crashed = None
    alt_rel = None
    last = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    spool_t0 = None
    while True:
        now = time.time()
        g = accel_g(link)
        # ---- commands
        if launch and phase in ("push", "fall"):
            send_target(link, *launch_cmd(link, now - t_trig, LAUNCH[kind], hover, t_rel is not None))
        elif script and phase in ("push", "fall"):
            ic.send_attitude(link, level_q(link), 0.5)       # still idle: nothing detected yet
        elif script and phase == "spool":
            ic.send_attitude(link, level_q(link), 1.0)       # climb-rate mode, positive = take-off -> spool up
        elif script and phase == "caught":
            ic.send_attitude(link, *height_hold(link, hover))   # caught: level, stop climbing / falling
        elif script and phase == "recover":
            if mode == "attitude-rule":
                ic.send_attitude(link, *ic.attitude_rule(link, hover))
            else:
                q, thr, last = ic.policy_action(mode, link, hover, alt_rel, last)
                ic.send_attitude(link, q, thr)
        rk.wait()
        link.update()
        g = accel_g(link)
        t = time.time()
        log_row(rows, n, kind, mode, phase, t - t_trig, link, spot, g)
        rate = float(np.linalg.norm(link.rates))
        peak_rate = max(peak_rate, rate)
        h = hgt()
        apex = max(apex, h)
        if t_rel is not None and (low is not None or h < apex - 0.05):
            low = h if low is None else min(low, h)                     # lowest point after the top
        if link.tilt_deg > 30:
            tilted = True
        # ---- release: first free-fall readings after the push (same rule for every mode)
        if t_rel is None and g is not None and t - t_trig > 0.05:
            n_ff = n_ff + 1 if g < FREEFALL_G else 0
            if n_ff >= FREEFALL_N:
                t_rel = t - (FREEFALL_N - 1) / HZ
                alt_rel = link.alt                         # (network input uses height CHANGE only)
                phase = "fall"
                if launch:
                    t_free_end = t + LAUNCH_FREE_S         # motors stay at minimum = hand-throw spool delay
                elif script:
                    phase, spool_t0 = "spool", t           # the script's detection = this moment
        # ---- motors at full range
        if t_motors is None and t_rel is not None:
            if launch and t >= t_free_end:
                t_motors = t                               # catch starts (motors were idling, not stopped)
                if ff is not None:
                    ic.send_params(link, {ic.FF_NAME: ff})   # your value back BEFORE the recovery
                phase = "recover"
            elif script and phase == "spool" and (landed_state(link) != LANDED_ON_GROUND or t - spool_t0 > SPOOL_MAX_S):
                t_motors = t
                if limits:
                    ic.send_params(link, {k: 0 for k in limits})          # rotation limits off (as the rules)
                ic.send_params(link, {"GUID_OPTIONS": int(guid) | ic.GUID_THRUST_BIT})   # thrust = real thrust
                phase = "recover"
            elif not script:
                for tt, text in link.texts:
                    if tt >= t_trig and "uprighting" in text:     # "throttle is unlimited - uprighting"
                        t_motors, phase = tt, "recover"
        if t_upright is None and tilted and t_motors is not None and link.tilt_deg < ic.TILT_OK:
            t_upright = t
        # ---- caught: same hand-back rule as impact_check (spinning stopped, upright, not falling fast)
        if script and phase == "recover" and rate < ic.RATE_OK and link.tilt_deg < ic.TILT_OK \
                and float(link.vel[2]) < 1.0:
            if limits:
                ic.send_params(link, limits)                 # stock rotation limits again
            phase, t_caught = "caught", t
        if not script and t_motors is not None and phase != "caught":
            for tt, text in link.texts:
                if tt >= t_trig and "controlling height" in text:  # Throw mode: "uprighted - controlling height"
                    phase, t_caught = "caught", tt
        # ---- calm?
        if t_motors is not None:
            ok = (rate < ic.RATE_OK and link.tilt_deg < ic.TILT_OK and h > ALT_OK
                  and float(np.linalg.norm(link.vel[0:2])) < ic.HSPEED_OK and abs(float(link.vel[2])) < VZ_OK)
            held = (held or t) if ok else None
            if held and t - held >= ic.HELD_S:
                t_rec = held
                break
        # ---- failures
        speed = hit_ground(link, t_trig)
        if speed is not None or (t_rel is not None and h < GROUND_M and float(link.vel[2]) > 1.0):
            crashed = speed if speed is not None else -1.0
            break
        if t_rel is None and t - t_trig > WAIT_MAX_S:
            break
        if t_rel is not None and t - t_rel > RECOVER_MAX_S:
            break
    # ---- hand back (if the loop ended before the catch)
    if script and phase == "recover" and limits:
        ic.send_params(link, limits)
    if launch and t_motors is None:                          # ended before the catch: put everything back
        if ff is not None:
            ic.send_params(link, {ic.FF_NAME: ff})
        if limits:
            ic.send_params(link, limits)
    drift = None
    hold_info = {}
    if t_rec is not None and crashed is None and after == "loiter":
        hold_info = loiter_hold(link, n, kind, mode, rows, t_trig, spot, hover, rk)
        drift = hold_info.pop("drift")
    elif t_rec is not None and crashed is None:
        hold = "GUIDED_NOGPS"
        try:                                                 # position hold (needs a position estimate)
            link.set_mode("GUIDED", timeout=3)               # mode change = ArduPilot initialises its controllers
            hold = "GUIDED"
        except RuntimeError:                                 # e.g. optical flow out of range: script height hold
            pass
        t_end = time.time() + HOLD_S
        while time.time() < t_end:
            if hold == "GUIDED":
                link.send_velocity(0.0, 0.0, 0.0)
            else:
                ic.send_attitude(link, *height_hold(link, hover))
            rk.wait()
            link.update()
            log_row(rows, n, kind, mode, "hold", time.time() - t_trig, link, spot, accel_g(link))
        drift = float(np.linalg.norm(link.pos[0:2] - spot[0:2]))
    if script:
        ic.send_params(link, {"GUID_OPTIONS": int(guid)})    # back to your value (no attitude commands follow)
    rel = (lambda x: round(x - t_rel, 2) if (x is not None and t_rel is not None) else None)
    info = {"n": n, "type": kind, "mode": mode,
            "released": t_rel is not None, "detected": t_motors is not None,
            "release_s": round(t_rel - t_trig, 2) if t_rel is not None else None,
            "motors_on_s": rel(t_motors), "upright_s": (rel(t_upright) if tilted else 0.0) if t_motors else None,
            "caught_s": rel(t_caught),
            "recovered": t_rec is not None and crashed is None, "recovered_s": rel(t_rec),
            "apex_m": round(apex, 2),
            "low_m": round(low if low is not None else apex, 2) if t_rel is not None else None,
            "peak_rate_rads": round(peak_rate, 2), "drift_m": round(drift, 2) if drift is not None else None,
            "crashed": crashed is not None, "impact_speed_ms": round(crashed, 2) if crashed and crashed > 0 else None}
    info.update(hold_info)
    info["hover_used"] = round(float(hover), 3)
    return info, crashed is not None


def main(args):
    default = ["attitude-rule"] if args.launch else ["throw-mode", "attitude-rule"]
    modes = list(args.modes or (default + (["ppo-policy"] if args.model else [])))
    if args.launch and "throw-mode" in modes:
        raise SystemExit("--launch (Gazebo) works with the script modes only: ArduPilot's Throw mode needs a real push "
                         "with the motors stopped. Use --modes attitude-rule ppo-policy")
    if "ppo-policy" in modes:
        if not args.model:
            raise SystemExit("ppo-policy needs --model, e.g. --model .\\models\\round1_final.zip")
        ic.register_policy("ppo-policy", args.model)
    nets = [m for m in modes if m in ("bc-policy", "ppo-policy")]
    if nets:
        ic.load_policies(nets)                               # BEFORE flying (torch import is slow)
    link = APLink(args.conn, rate_hz=50)
    link.wait_state()
    link.request_rate(M.MAVLINK_MSG_ID_RAW_IMU, 50)
    link.request_rate(M.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 10)
    link.request_rate(M.MAVLINK_MSG_ID_VFR_HUD, 10)        # --launch: hover thrust is measured from it
    if not wait_origin(link):
        raise SystemExit("No EKF origin after 60 s - is the EKF running? (check the SITL console)")
    t_end = time.time() + 5
    while time.time() < t_end and not {"RAW_IMU", "EXTENDED_SYS_STATE"} <= set(link.msgs):
        link.update()
        time.sleep(0.05)
    missing = {"RAW_IMU", "EXTENDED_SYS_STATE"} - set(link.msgs)
    if missing:
        raise SystemExit(f"The FC does not send {', '.join(sorted(missing))} - needed for throw detection")
    got = link.get_params(["SIM_SHOVE_TIME", "EK3_SRC1_POSXY", "EK3_SRC1_VELXY", "GUID_OPTIONS", "MOT_THST_HOVER"])
    if "SIM_SHOVE_TIME" not in got:
        raise SystemExit("No SIM_SHOVE_TIME: this is not ArduPilot's built-in simulator. "
                         "Start it with: sim_vehicle.py -v ArduCopter -f quad --console --map (no Gazebo)")
    guid = int(got.get("GUID_OPTIONS", 0))
    hover = float(got.get("MOT_THST_HOVER", 0.4))
    limits = ic.read_limits(link)
    t_end = time.time() + 3                                  # armed state comes with the FC heartbeat (1 Hz);
    while time.time() < t_end and link.mode == "?":          # the parameter reads above skip heartbeats
        link.update()
        time.sleep(0.05)
    if link.armed and not make_disarmed(link, "left from an earlier run"):
        raise SystemExit("STOP - the drone is still ARMED and could not be disarmed. Restart sim_vehicle")
    if (guid & ic.GUID_THRUST_BIT) or any(abs(v) < 1e-6 for v in limits.values()):
        # test values left from a run that was stopped mid-throw -> put ArduPilot's values back
        fix = {"GUID_OPTIONS": guid & ~ic.GUID_THRUST_BIT}
        for k, v in limits.items():
            if abs(v) < 1e-6:
                if k not in STOCK_LIMITS:                    # newer firmware names: units not checked
                    raise SystemExit(f"STOP - {k} = 0 left from a crashed run: set it back in MAVProxy")
                fix[k] = STOCK_LIMITS[k]
        print(f"[sim] test values left from a stopped run - restored: {fix}")
        ic.send_params(link, fix)
        guid = fix["GUID_OPTIONS"]
        limits = {k: fix.get(k, v) for k, v in limits.items()}
    ff = None
    if args.launch:                                          # lead spin kick: ATC_RATE_FF_ENAB 0 during the kick only
        ff = link.get_params([ic.FF_NAME]).get(ic.FF_NAME)
        if ff is not None and int(ff) == 0:
            print(f"[sim] {ic.FF_NAME} = 0 left from a stopped run - restored to 1 (ArduPilot default)")
            ic.send_params(link, {ic.FF_NAME: 1})
            ff = 1.0
    pos_src, vel_src = int(got.get("EK3_SRC1_POSXY", -1)), int(got.get("EK3_SRC1_VELXY", -1))
    flow = vel_src == 5                                      # optical flow: fly back low (rangefinder range)
    after = "fly-back" if args.fly_back else "loiter"        # what happens after a recovery (section 8)
    src = ("optical flow (relative position only)" if pos_src == 0 and vel_src == 5 else
           f"{SRC_NAMES.get(pos_src, '?')} position, {SRC_NAMES.get(vel_src, '?')} velocity")
    print(f"Position source: {src}  |  hover thrust {hover:.2f}  |  modes {', '.join(modes)}")
    if args.launch:
        print(f"   LAUNCH mode (Gazebo): climb to {LAUNCH_ALT:.0f} m, push {LAUNCH_PUSH_S} s, spin kick, release; "
              f"motors stay at minimum {LAUNCH_FREE_S} s, then the catch ('motors on' = that delay)")
    if "throw-mode" in modes and int(got.get("EK3_SRC1_POSXY", 3)) != 3:
        print("   NOTE: no GPS position -> ArduPilot Throw mode cannot detect the throw (mode_throw.cpp needs an "
              "absolute position). Expect throw-mode = not detected / crash. That is the result to record.")
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / f"throw_check_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results, rows, n = [], [], 0
    stop = None
    try:
        for kind in args.types:
            for _ in range(args.repeats):
                for mode in modes:
                    n += 1
                    e, crash = one_throw(link, kind, mode, n, rows, hover, guid, limits, after, args.launch, ff)
                    results.append(e)
                    fmt = (lambda v, f="{:.1f}": "-" if v is None else f.format(v))
                    res = (e["note"] if e.get("note") and not e["recovered"] else
                           f"recovered {e['recovered_s']:.2f}s after release" if e["recovered"] else
                           f"CRASHED at {e['impact_speed_ms']} m/s" if e["crashed"] and e["impact_speed_ms"] else
                           "CRASHED" if e["crashed"] else "not recovered")
                    print(f"Throw {n:2d} | {kind:<5} | {mode:<13} | detected {'yes' if e['detected'] else 'NO '}"
                          f" | motors on {fmt(e['motors_on_s'], '{:.2f}'):>5} s"
                          f" | top {fmt(e['apex_m'])} m, lowest {fmt(e['low_m'])} m"
                          f" | peak rate {fmt(e['peak_rate_rads'])} | drift {fmt(e['drift_m'])} m | {res}")
                    if e.get("hold_mode"):
                        print(f"         hold: {e['hold_mode']} {e['hold_s']} s | braked {e.get('brake_m')} m in "
                              f"{e.get('brake_s')} s | then held within {e.get('wander_m')} m | height +/-{e['hold_dz_m']} m"
                              + (f" | {e['note']}" if e.get("note") else ""))
                    if crash:
                        force_disarm(link)
                        if link.armed or link.tilt_deg > 30:         # on its side / still armed: SITL unreliable
                            stop = "crash"
                            raise StopIteration
                        print("   (SITL drone landed upright after the crash - disarmed, continuing)")
                        continue
                    land_and_disarm(link, flow, go_home=(after == "fly-back"))
    except StopIteration:
        print("\nThe drone hit the ground - results saved. Restart sim_vehicle (Ctrl+C, same command) "
              "before the next run.")
    except KeyboardInterrupt:
        print("\nStopped by user - saving what we have")
        stop = "user"
    finally:
        ic.send_params(link, {"GUID_OPTIONS": guid})
        if ff is not None:
            ic.send_params(link, {ic.FF_NAME: ff})
        if limits:
            ic.send_params(link, limits)
        with open(f"{base}.csv", "w", newline="") as f:
            keys = list(dict.fromkeys(k for e in results for k in e)) or ["n"]
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        with open(f"{base}_steps.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "type", "mode", "phase", "t_s", "tilt_deg", "rate_rads", "alt_m", "hspeed_ms",
                        "dist_from_spot_m", "accel_g", "landed_state", "p_rads", "q_rads", "r_rads"])
            w.writerows(rows)
        summary = {}
        for mode in modes:
            for kind in args.types:
                ee = [e for e in results if e["mode"] == mode and e["type"] == kind]
                if ee:
                    rec = [e for e in ee if e["recovered"]]
                    summary[f"{mode}/{kind}"] = {
                        "recovered": f"{len(rec)}/{len(ee)}",
                        "mean_recovered_s": round(float(np.mean([e["recovered_s"] for e in rec])), 2) if rec else None,
                        "mean_low_m": round(float(np.mean([e["low_m"] for e in rec if e["low_m"] is not None])), 2)
                        if any(e["low_m"] is not None for e in rec) else None,
                        "mean_drift_m": round(float(np.mean([e["drift_m"] for e in rec if e["drift_m"] is not None])), 2)
                        if any(e["drift_m"] is not None for e in rec) else None,
                        "mean_brake_m": round(float(np.mean([e["brake_m"] for e in rec if e.get("brake_m") is not None])), 2)
                        if any(e.get("brake_m") is not None for e in rec) else None,
                        "mean_wander_m": round(float(np.mean([e["wander_m"] for e in rec if e.get("wander_m") is not None])), 2)
                        if any(e.get("wander_m") is not None for e in rec) else None}
        json.dump({"date": datetime.datetime.now().isoformat(timespec="seconds"), "position_source": src,
                   "hover": hover, "modes": modes, "types": args.types, "repeats": args.repeats,
                   "throws": THROWS, "freefall_g": FREEFALL_G, "freefall_n": FREEFALL_N, "model": args.model,
                   "after_recovery": after, "launch": bool(args.launch), "loiter_s": LOITER_S, "stopped": stop, "summary": summary, "results": results},
                  open(f"{base}.json", "w"), indent=2)
        print("\n" + "=" * 70)
        for k, v in summary.items():
            print(f"{k:<28} recovered {v['recovered']:<5} | mean time {v['mean_recovered_s']} s | "
                  f"mean lowest {v['mean_low_m']} m | mean drift {v['mean_drift_m']} m"
                  + (f" | braked {v['mean_brake_m']} m, held within {v['mean_wander_m']} m"
                     if v['mean_wander_m'] is not None else ""))
        print(f"\nSaved: {base}.csv / _steps.csv / .json")
        if stop is None:
            land_and_disarm(link, flow, go_home=(after == "fly-back"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["up", "roll"], choices=list(THROWS))
    ap.add_argument("--modes", nargs="+", default=None, choices=list(MODES))
    ap.add_argument("--model", default=None, help="Stage 3 PPO .zip -> adds mode ppo-policy")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    ap.add_argument("--launch", action="store_true",
                    help="Gazebo: the drone throws itself (climb, push, spin, release) - no SIM_SHOVE needed")
    ap.add_argument("--fly-back", action="store_true",
                    help="old ending: GUIDED hold 3 s, fly back to the throw spot (default: LOITER where it recovered)")
    main(ap.parse_args())
