"""
WIND CHECK (one-off diagnostic, not part of the Stage 2 pipeline)
=================================================================
Question: does AirSim's wind (simSetWind) actually push the ArduPilot drone?

To hold position against wind, a drone MUST lean into it. So we compare the tilt:
    Part 1  no wind,             hold position 8 s   -> tilt should be ~0-2 deg
    Part 2  5 m/s from the north, hold position 8 s  -> tilt should rise clearly if wind works
    Part 3  5 m/s, commands = zero velocity only, 4 s -> how far it gets pushed
Then wind off, land.

Run (AirSim + SITL in GPS mode, from the Stage_2 folder):
    py .\\wind_check.py               (5 m/s)
    py .\\wind_check.py --wind 15     (any strength)
"""

import argparse
import math

import airsim
import numpy as np

from ap_link import APLink, RateKeeper
from pid_expert import PIDExpertHover

HZ = 20.0


def lean(q):
    """Roll and pitch in degrees from quaternion w, x, y, z."""
    w, x, y, z = q
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    return roll, pitch


def phase(link, sim, target, teacher, seconds, wind, hold=True, label=""):
    sim.simSetWind(airsim.Vector3r(float(wind[0]), float(wind[1]), float(wind[2])))
    rk = RateKeeper(HZ)
    tilt, pitch, roll, dist = [], [], [], []
    for _ in range(int(seconds * HZ)):
        link.update()
        o = link.obs(target)
        if hold:
            link.send_velocity(*teacher.get_action({"position": o[0:3], "velocity": o[3:6]}))
        else:
            link.send_velocity(0.0, 0.0, 0.0)
        r, p = lean(link.q)
        tilt.append(link.tilt_deg)
        roll.append(r)
        pitch.append(p)
        dist.append(float(np.linalg.norm(o[0:2])))
        rk.wait()
    half = len(tilt) // 2                       # second half = after it has reacted
    res = {"tilt": float(np.mean(tilt[half:])), "pitch": float(np.mean(pitch[half:])),
           "roll": float(np.mean(roll[half:])), "max_dist": float(max(dist)), "end_dist": float(dist[-1])}
    print(f"  {label:<40} tilt {res['tilt']:5.1f} deg (pitch {res['pitch']:+5.1f}, roll {res['roll']:+5.1f}) | "
          f"max drift {res['max_dist']:.2f} m | end {res['end_dist']:.2f} m")
    return res


def main(WIND):
    sim = airsim.MultirotorClient()
    sim.confirmConnection()
    sim.simSetWind(airsim.Vector3r(0, 0, 0))
    link = APLink()
    link.wait_state()
    if not link.armed:
        link.takeoff_nogps(2.0)
    else:
        link.set_mode("GUIDED")
    link.update()
    target = np.array([link.pos[0], link.pos[1], -2.0])
    teacher = PIDExpertHover()
    print(f"\n=== WIND CHECK: {WIND} m/s wind from the north (blowing south) ===\n")
    try:
        a = phase(link, sim, target, teacher, 8, [0, 0, 0], label="Part 1: no wind, holding")
        b = phase(link, sim, target, teacher, 8, [-WIND, 0, 0], label=f"Part 2: {WIND} m/s wind, holding")
        phase(link, sim, target, teacher, 4, [-WIND, 0, 0], hold=False,
                  label=f"Part 3: {WIND} m/s wind, zero-velocity cmd")
        phase(link, sim, target, teacher, 6, [0, 0, 0], label="Wind off, back to target")
    finally:
        sim.simSetWind(airsim.Vector3r(0, 0, 0))
        link.hold()
    dp, dr = b["pitch"] - a["pitch"], b["roll"] - a["roll"]
    lean_change = math.hypot(dp, dr)
    print("\nVERDICT")
    print(f"  Lean change with {WIND} m/s wind: {lean_change:.2f} deg (pitch {dp:+.2f}, roll {dr:+.2f}), "
          f"tilt {a['tilt']:.2f} -> {b['tilt']:.2f} deg")
    if lean_change > 0.3:
        print(f"  WIND WORKS: the drone leans {lean_change:.1f} deg into the wind to hold position.")
        if lean_change < 3.0:
            print("  But the push is SMALL: ArduPilot cancels it easily (max drift "
                  f"{b['max_dist']:.2f} m). AirSim wind only adds air drag, and the default drone has little drag.")
    else:
        print(f"  WIND DOES NOT REACH THE DRONE: lean changed only {lean_change:.2f} deg at {WIND} m/s.")
    link.land()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wind", type=float, default=5.0, help="wind strength in m/s")
    main(ap.parse_args().wind)
