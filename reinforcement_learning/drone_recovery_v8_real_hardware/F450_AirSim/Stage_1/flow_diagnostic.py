"""
FLOW DIAGNOSTIC (one-off investigation, not part of the Stage 1 pipeline)
=========================================================================
Question: why does the drone drift in optical-flow mode but not in GPS mode?

It compares two things at 20 Hz:
    ArduPilot's estimate  (what the teacher / network see)    - via MAVLink
    AirSim's true pose    (where the drone REALLY is)          - via AirSim API

Test (about 2 minutes, drone stays within ~1.5 m of the start):
    Phase A  hold still 40 s            -> does the estimate drift? does the real drone drift?
    Phase B  0.5 m/s north / east / south / west, 3 s each, 3 s stop between
                                          -> does the drone move the way it is told?
Then it lands and prints a verdict.

Run it ONCE in GPS mode and ONCE in optical-flow mode (same AirSim + SITL setup as Stage 1):
    py .\\flow_diagnostic.py
Saves: runs\\diagnostic\\flow_diagnostic_<sensor>_<date>.csv  (+ .txt with the verdict)
"""

import csv
import datetime
import math
from pathlib import Path

import airsim
import numpy as np

from ap_link import APLink, RateKeeper

HZ = 20.0
ALT = 2.0
SPEED = 0.5
TRUE_FENCE_M = 3.0          # real distance from start -> stop and land


def yaw_deg_from_q(w, x, y, z):
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def main():
    link = APLink()
    link.wait_state()
    params = link.get_params(["EK3_SRC1_POSXY", "EK3_SRC1_VELXY"])
    sensor = "optflow" if params.get("EK3_SRC1_VELXY") == 5 else \
        ("gps" if params.get("EK3_SRC1_POSXY") == 3 else "unknown")

    sim = airsim.MultirotorClient()
    sim.confirmConnection()

    def truth():
        p = sim.simGetVehiclePose()
        q = p.orientation
        return (np.array([p.position.x_val, p.position.y_val, p.position.z_val]),
                yaw_deg_from_q(q.w_val, q.x_val, q.y_val, q.z_val))

    out_dir = Path("runs") / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"flow_diagnostic_{sensor}_{stamp}"
    print(f"\n=== FLOW DIAGNOSTIC | sensor mode: {sensor} | saving {base}.csv ===\n")

    z_ground = None
    if not link.armed:
        z_ground = truth()[0][2]          # AirSim z while standing on the floor
        print(f"AirSim z on the ground: {z_ground:+.2f} m (start point is that far above the floor)")
        link.takeoff_nogps(ALT)
    else:
        link.set_mode("GUIDED")

    link.update()
    est0 = link.pos.copy()
    true0, _ = truth()

    plan = [("hold", np.zeros(3), 40.0)]
    for name, v in (("north", [SPEED, 0, 0]), ("east", [0, SPEED, 0]),
                    ("south", [-SPEED, 0, 0]), ("west", [0, -SPEED, 0])):
        plan += [(name, np.array(v, float), 3.0), (f"stop_after_{name}", np.zeros(3), 3.0)]

    rows = []
    rk = RateKeeper(HZ)
    t = 0.0
    stopped = None
    try:
        for phase, cmd, dur in plan:
            print(f"  phase {phase:16s} command N {cmd[0]:+.1f} E {cmd[1]:+.1f} m/s for {dur:.0f} s")
            for _ in range(int(dur * HZ)):
                link.update()
                tp, tyaw = truth()
                est_rel = link.pos - est0
                true_rel = tp - true0
                eyaw = yaw_deg_from_q(*link.q)
                rows.append([round(t, 3), phase, cmd[0], cmd[1],
                             est_rel[0], est_rel[1], -link.pos[2], link.vel[0], link.vel[1],
                             true_rel[0], true_rel[1],
                             (z_ground - tp[2]) if z_ground is not None else float("nan"),
                             eyaw, tyaw, link.tilt_deg])
                if np.linalg.norm(true_rel[0:2]) > TRUE_FENCE_M:
                    stopped = f"real drone moved > {TRUE_FENCE_M} m from start during '{phase}'"
                    raise StopIteration
                link.send_velocity(*cmd)
                rk.wait()
                t += 1.0 / HZ
    except (StopIteration, KeyboardInterrupt):
        pass
    finally:
        link.hold()

    cols = ["t", "phase", "cmd_n", "cmd_e", "est_n", "est_e", "est_alt", "est_vn", "est_ve",
            "true_n", "true_e", "true_alt", "est_yaw", "true_yaw", "tilt"]
    with open(f"{base}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    # ------------------------------------------------------------- analysis
    R = {c: np.array([r[i] for r in rows]) if c != "phase" else [r[i] for r in rows]
         for i, c in enumerate(cols)}
    phase = np.array(R["phase"])
    L = [f"FLOW DIAGNOSTIC - sensor mode: {sensor} - {stamp}", "=" * 72]
    if stopped:
        L.append(f"STOPPED EARLY: {stopped}")

    # heading
    yaw_err = wrap(R["est_yaw"] - R["true_yaw"])
    L.append(f"Heading: ArduPilot minus AirSim = {np.mean(yaw_err):+.1f} deg on average "
             f"(range {yaw_err.min():+.1f} .. {yaw_err.max():+.1f})")

    # hold phase drift
    h = phase == "hold"
    if h.any():
        est_d = np.hypot(R["est_n"][h][-1] - R["est_n"][h][0], R["est_e"][h][-1] - R["est_e"][h][0])
        true_d = np.hypot(R["true_n"][h][-1] - R["true_n"][h][0], R["true_e"][h][-1] - R["true_e"][h][0])
        gap = np.hypot((R["est_n"] - R["true_n"])[h][-1], (R["est_e"] - R["true_e"])[h][-1])
        dur = R["t"][h][-1] - R["t"][h][0]
        L += ["", f"PHASE A - hold still {dur:.0f} s",
              f"  ArduPilot thinks it moved: {est_d:.2f} m",
              f"  Drone REALLY moved:        {true_d:.2f} m",
              f"  Gap estimate vs real at the end: {gap:.2f} m",
              f"  Height above floor: ArduPilot {R['est_alt'][h][-1]:.2f} m | real {R['true_alt'][h][-1]:.2f} m"]

    # step phases: true motion direction and speed vs command
    L += ["", "PHASE B - 0.5 m/s steps (real movement vs command)",
          f"  {'step':6s} {'real dist':>9s} {'real dir':>9s} {'cmd dir':>8s} {'dir error':>9s} "
          f"{'est dist':>9s} {'est/real':>8s}"]
    dir_errs, ratios = [], []
    for name, ang in (("north", 0.0), ("east", 90.0), ("south", 180.0), ("west", -90.0)):
        m = phase == name
        if m.sum() < 5:
            continue
        dn = R["true_n"][m][-1] - R["true_n"][m][0]
        de = R["true_e"][m][-1] - R["true_e"][m][0]
        en = R["est_n"][m][-1] - R["est_n"][m][0]
        ee = R["est_e"][m][-1] - R["est_e"][m][0]
        real, est = math.hypot(dn, de), math.hypot(en, ee)
        rdir = math.degrees(math.atan2(de, dn))
        derr = wrap(rdir - ang)
        dir_errs.append(derr)
        ratios.append(est / real if real > 1e-3 else float("nan"))
        L.append(f"  {name:6s} {real:8.2f}m {rdir:+8.0f}° {ang:+7.0f}° {derr:+8.0f}° {est:8.2f}m "
                 f"{ratios[-1]:8.2f}")

    # verdict
    L += ["", "VERDICT"]
    if dir_errs:
        md = float(np.mean(np.abs(dir_errs)))
        mr = float(np.nanmean(ratios))
        if md > 20:
            L.append(f"  DIRECTION PROBLEM: the drone moves {md:.0f} deg away from the commanded direction "
                     "-> ArduPilot's heading is wrong (compass / yaw). This makes a controller spiral away.")
        if not 0.8 <= mr <= 1.25:
            L.append(f"  SCALE PROBLEM: ArduPilot's distance is {mr:.2f} x the real distance "
                     "-> optical-flow speed is scaled wrong (height / flow scaling).")
        if md <= 20 and 0.8 <= mr <= 1.25:
            L.append(f"  Commands work: direction error {md:.0f} deg, distance ratio {mr:.2f}.")
    if h.any() and gap > 0.3:
        L.append(f"  POSITION DRIFT: estimate and reality were {gap:.2f} m apart after holding still "
                 f"{dur:.0f} s -> the position estimate drifts.")
    if stopped:
        L.append("  The real drone left the 3 m test area - see the CSV for when.")
    text = "\n".join(L)
    print("\n" + text)
    with open(f"{base}.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSaved: {base}.csv / .txt")
    link.land()


if __name__ == "__main__":
    main()
