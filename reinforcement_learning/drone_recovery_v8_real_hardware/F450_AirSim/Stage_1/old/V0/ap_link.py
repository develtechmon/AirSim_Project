"""
ap_link.py - MAVLink link to ArduCopter. ONE file for SITL, HITL and the real drone.
=====================================================================================
Only the connection string changes:
    SITL (direct, 50 Hz OK) : tcp:127.0.0.1:5762
    Real drone (RPi UART)   : /dev/serial0   (with --baud 921600, set later)

Observation (13) - same size as your old pipeline, but altitude-independent:
    [0:3]   rel_pos  = position - target       (NED, m)   <- was absolute position
    [3:6]   velocity                            (NED, m/s)
    [6:10]  quaternion w, x, y, z
    [10:13] body rates p, q, r                  (rad/s)

Commands:
    send_velocity(vx, vy, vz)   Stage 1 & 2  (GUIDED, NED m/s, yaw held)
    send_rates(p, q, r, thrust) Stage 3      (GUIDED, rates-only - verified by rate_test.py)
"""

import math
import time

import numpy as np
from pymavlink import mavutil

M = mavutil.mavlink

# SET_POSITION_TARGET_LOCAL_NED mask: use velocity + yaw_rate, ignore everything else
VEL_MASK = (M.POSITION_TARGET_TYPEMASK_X_IGNORE | M.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            M.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            M.POSITION_TARGET_TYPEMASK_AX_IGNORE | M.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            M.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            M.POSITION_TARGET_TYPEMASK_YAW_IGNORE)
RATES_MASK = 0b10000000          # SET_ATTITUDE_TARGET: ignore quaternion, use rates + thrust

JUMP_M = 0.5        # horizontal step between two position messages (20 ms apart) that cannot be
                    # real motion -> EKF position reset; it is removed so position stays smooth
SETTLE_S = 15.0     # max wait after GUIDED for the EKF heading to align before flying


class APLink:
    def __init__(self, conn="tcp:127.0.0.1:5762", baud=115200, rate_hz=50):
        print(f"[link] connecting {conn}")
        self.m = mavutil.mavlink_connection(conn, baud=baud, source_system=255)
        if self.m.wait_heartbeat(timeout=30) is None:
            raise RuntimeError("no heartbeat - is SITL / the FC running?")
        print(f"[link] heartbeat from system {self.m.target_system}")

        self.pos = self.vel = self.q = self.rates = None
        self.armed = False
        self.mode = "?"
        self._last_raw = None              # last raw position from the FC
        self._offset = np.zeros(3)         # sum of EKF reset jumps removed so far
        self.yaw_aligned = False           # set when the FC reports yaw alignment complete
        self.set_ekf_origin()
        for msg_id in (M.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
                       M.MAVLINK_MSG_ID_ATTITUDE_QUATERNION):
            self.m.mav.command_long_send(
                self.m.target_system, self.m.target_component,
                M.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, int(1e6 / rate_hz), 0, 0, 0, 0, 0)

    # ---------------------------------------------------------------- state
    def update(self):
        """Drain all waiting messages (non-blocking)."""
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                return
            t = msg.get_type()
            if t == "LOCAL_POSITION_NED":
                raw = np.array([msg.x, msg.y, msg.z])
                if self._last_raw is not None:
                    jump = raw - self._last_raw
                    if np.linalg.norm(jump[0:2]) > JUMP_M:      # EKF reset, not real motion
                        self._offset[0:2] -= jump[0:2]
                        print(f"[link] EKF position reset of {np.linalg.norm(jump[0:2]):.2f} m "
                              f"removed (position kept smooth)")
                self._last_raw = raw
                self.pos = raw + self._offset
                self.vel = np.array([msg.vx, msg.vy, msg.vz])
            elif t == "ATTITUDE_QUATERNION":
                self.q = np.array([msg.q1, msg.q2, msg.q3, msg.q4])
                self.rates = np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed])
            elif t == "HEARTBEAT" and msg.get_srcSystem() == self.m.target_system \
                    and msg.type != M.MAV_TYPE_GCS:
                self.armed = bool(msg.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
                self.mode = self.m.flightmode
            elif t == "STATUSTEXT":
                print(f"[fc] {msg.text}")
                if "yaw alignment complete" in msg.text.lower():
                    self.yaw_aligned = True

    def ready(self):
        return all(v is not None for v in (self.pos, self.vel, self.q, self.rates))

    def wait_state(self, timeout=30):
        t_end = time.time() + timeout
        while time.time() < t_end:
            self.update()
            if self.ready():
                return
            time.sleep(0.02)
        raise RuntimeError("no LOCAL_POSITION_NED / ATTITUDE_QUATERNION - EKF not ready?")

    @property
    def alt(self):
        return -self.pos[2]

    @property
    def tilt_deg(self):
        w, x, y, z = self.q
        up_z = 1.0 - 2.0 * (x * x + y * y)
        return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))

    def obs(self, target):
        """13-dim observation. target = NED position [x, y, z] (z negative = up)."""
        return np.concatenate([self.pos - target, self.vel, self.q, self.rates]).astype(np.float32)

    # ------------------------------------------------------------- commands
    def set_ekf_origin(self, lat=-35.363261, lon=149.165230, alt_m=584.0):
        """Without GPS, ArduPilot needs an EKF origin before it gives a position.
        Safe to send every time: ignored if the origin is already set (e.g. GPS mode)."""
        self.m.mav.set_gps_global_origin_send(
            self.m.target_system, int(lat * 1e7), int(lon * 1e7), int(alt_m * 1000))

    def set_mode(self, name, timeout=10):
        self.m.set_mode(name)
        t_end = time.time() + timeout
        while time.time() < t_end:
            self.update()
            if self.mode == name:
                print(f"[link] mode {name}")
                return
            time.sleep(0.05)
        raise RuntimeError(f"mode {name} not accepted (now {self.mode})")

    def arm(self, retries=10):
        for _ in range(retries):
            self.m.arducopter_arm()
            t_end = time.time() + 3
            while time.time() < t_end:
                self.update()
                if self.armed:
                    print("[link] armed")
                    return
                time.sleep(0.05)
        raise RuntimeError("arming failed - check pre-arm messages above")

    def takeoff(self, alt, timeout=60):
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            M.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt)
        t_end = time.time() + timeout
        while time.time() < t_end:
            self.update()
            if self.pos is not None and self.alt >= 0.95 * alt:
                print(f"[link] takeoff done, alt {self.alt:.2f} m")
                return
            time.sleep(0.1)
        raise RuntimeError("takeoff timeout")

    def takeoff_nogps(self, alt, timeout=40):
        """Indoor takeoff (works with optical flow AND GPS, and on the real drone):
        1. arm + climb in GUIDED_NOGPS (no position needed)
        2. hover until the EKF has a position from optical flow
        3. switch to GUIDED"""
        self.update()
        w, x, y, z = self.q
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        q_yaw = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]   # level, keep heading

        def level(thrust):   # thrust = climb rate: 0.5 hold, >0.5 climb
            self.m.mav.set_attitude_target_send(
                0, self.m.target_system, self.m.target_component,
                0b00000111, q_yaw, 0, 0, 0, float(thrust))

        self.set_mode("GUIDED_NOGPS")
        self.arm()
        print(f"[link] climbing to {alt:.1f} m (GUIDED_NOGPS)")
        t_end = time.time() + timeout
        while time.time() < t_end:
            self.update()
            if self.alt >= 0.95 * alt:
                break
            level(0.7)
            time.sleep(0.05)
        else:
            raise RuntimeError("takeoff timeout")
        print(f"[link] at {self.alt:.2f} m - waiting for position estimate, then GUIDED")

        t_end = time.time() + 30
        while time.time() < t_end:
            self.m.set_mode("GUIDED")
            t_try = time.time() + 1.0
            while time.time() < t_try:
                level(0.5)
                self.update()
                if self.mode == "GUIDED":
                    print("[link] mode GUIDED")
                    self.settle()
                    return
                time.sleep(0.05)
        raise RuntimeError("GUIDED not accepted - no position estimate from optical flow")

    def settle(self, max_s=SETTLE_S, after_s=3.0):
        """Hold still until the EKF heading is aligned (+ a few seconds), so the first
        velocity commands go in the right direction. Real drone: same message from the FC."""
        print("[link] holding still - waiting for EKF yaw alignment")
        t_end = time.time() + max_s
        while time.time() < t_end and not self.yaw_aligned:
            self.update()
            self.hold()
            time.sleep(0.05)
        if not self.yaw_aligned:
            print("[link] no yaw-alignment message seen - continuing (heading may already be aligned)")
        t_end = time.time() + after_s
        while time.time() < t_end:
            self.update()
            self.hold()
            time.sleep(0.05)
        print("[link] settled - ready to fly")

    def send_velocity(self, vx, vy, vz, yaw_rate=0.0):
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            M.MAV_FRAME_LOCAL_NED, VEL_MASK,
            0, 0, 0, float(vx), float(vy), float(vz), 0, 0, 0, 0, float(yaw_rate))

    def send_rates(self, p, q, r, thrust):
        self.m.mav.set_attitude_target_send(
            0, self.m.target_system, self.m.target_component,
            RATES_MASK, [1.0, 0.0, 0.0, 0.0], float(p), float(q), float(r), float(thrust))

    def hold(self):
        self.send_velocity(0.0, 0.0, 0.0)

    # ------------------------------------------------------- record keeping
    def firmware_version(self, timeout=3):
        """Read the firmware version from the FC itself (for the run record)."""
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            M.MAV_CMD_REQUEST_MESSAGE, 0,
            M.MAVLINK_MSG_ID_AUTOPILOT_VERSION, 0, 0, 0, 0, 0, 0)
        msg = self.m.recv_match(type="AUTOPILOT_VERSION", blocking=True, timeout=timeout)
        if msg is None:
            return "unknown"
        v = msg.flight_sw_version
        kind = {0: "dev", 64: "alpha", 128: "beta", 192: "rc", 255: "official"}.get(v & 255, v & 255)
        return f"{(v >> 24) & 255}.{(v >> 16) & 255}.{(v >> 8) & 255}-{kind}"

    def get_params(self, names, timeout=5):
        """Read parameter values from the FC. Names that don't exist are skipped."""
        out = {}
        for n in names:
            self.m.mav.param_request_read_send(
                self.m.target_system, self.m.target_component, n.encode(), -1)
        t_end = time.time() + timeout
        while time.time() < t_end and len(out) < len(names):
            msg = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if msg is None:
                continue
            pid = msg.param_id
            if isinstance(pid, bytes):
                pid = pid.decode(errors="ignore")
            pid = pid.rstrip("\x00")
            if pid in names:
                out[pid] = msg.param_value
        return out


class RateKeeper:
    """Fixed-rate loop. Records real step times so the thesis reports measured seconds."""

    def __init__(self, hz):
        self.dt = 1.0 / hz
        self.next_t = time.perf_counter()
        self.last = None
        self.dts = []
        self.overruns = 0

    def wait(self):
        self.next_t += self.dt
        now = time.perf_counter()
        if self.next_t > now:
            time.sleep(self.next_t - now)
        else:
            self.overruns += 1
            self.next_t = now
        t = time.perf_counter()
        if self.last is not None:
            self.dts.append(t - self.last)
        self.last = t

    def stats(self):
        d = np.array(self.dts) if self.dts else np.array([0.0])
        return {"dt_mean": float(d.mean()), "dt_std": float(d.std()),
                "dt_max": float(d.max()), "overruns": self.overruns}
