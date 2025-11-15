"""
AIRSIM RECOVERY ENVIRONMENT + PID TEACHER (FIXED RETURNS, SHAPES, & RPC TYPES)
Author: Lukas the Big Boss + GPT-5

- TeacherPD: PD hover teacher (rates + throttle)
- PostImpactRecoveryEnvRate: Gym env with teacher (rates) or agent (PWM)
"""

import time
from collections import deque
from typing import Tuple, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import airsim


# ---------------------------
# Utilities
# ---------------------------

def to_float(x):
    """Convert numpy scalars to native Python float."""
    if isinstance(x, (np.floating,)):
        return float(x)
    return x

def clamp01(x):
    return np.clip(x, 0.0, 1.0)


# ---------------------------
# Teacher PD (rates + throttle)
# ---------------------------

class TeacherPD:
    """
    Simple PD hover teacher controlling roll/pitch/yaw *rates* + throttle.
    Used only for *data collection / imitation teacher*.
    """

    def __init__(self, alt_target: float = 15.0):
        # Attitude gains (gentle)
        self.kp_roll = 0.6
        self.kd_roll = 0.25
        self.kp_pitch = 0.6
        self.kd_pitch = 0.25
        self.kp_yaw = 0.25
        self.kd_yaw = 0.1

        # Altitude loop (outer)
        self.kp_z = 0.45
        self.kd_z = 0.20

        self.alt_target = alt_target

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Input: observation (expects env's 34D layout)
        Output: np.array([roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd, throttle])
        """
        # Extract signals from obs
        ang_vel = obs[3:6]         # wx, wy, wz
        roll, pitch, yaw = obs[12:15]
        altitude = obs[15]
        vertical_vel = obs[16]

        # PD in angle domain (rates as dampers)
        roll_rate_cmd  = -self.kp_roll  * roll  - self.kd_roll  * ang_vel[0]
        pitch_rate_cmd = -self.kp_pitch * pitch - self.kd_pitch * ang_vel[1]
        yaw_rate_cmd   = -self.kp_yaw   * yaw   - self.kd_yaw   * ang_vel[2]

        # Altitude PD
        z_err = (self.alt_target - altitude)
        throttle = 0.67 + 0.05 * (self.kp_z * z_err - self.kd_z * vertical_vel)

        # Clip
        roll_rate_cmd  = np.clip(roll_rate_cmd,  -0.6, 0.6)
        pitch_rate_cmd = np.clip(pitch_rate_cmd, -0.6, 0.6)
        yaw_rate_cmd   = np.clip(yaw_rate_cmd,   -0.6, 0.6)
        throttle       = np.clip(throttle,        0.50, 0.80)

        return np.array([roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd, throttle], dtype=np.float32)


# ---------------------------
# Environment (rates -> teacher, PWM -> agent)
# ---------------------------

class PostImpactRecoveryEnvRate(gym.Env):
    """
    AirSim Environment (Continuous)
      - Teacher mode: uses moveByAngleRatesThrottleAsync (stable)
      - Agent mode:   uses moveByMotorPWMsAsync (raw PWM)

    Action (4):
      [roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd, throttle] in [-1,1] (rates)
      When in PWM mode, we still pass the same action vector and map to PWM internally.

    Observation (34):
      [0:3]   linear vel (x,y,z)
      [3:6]   angular vel (x,y,z)
      [6:10]  orientation quat (w,x,y,z)
      [10:13] linear acc (x,y,z)
      [13:16] euler (roll, pitch, yaw)
      [16]    altitude
      [17]    vertical velocity (+up)
      [18:22] prev action (4)
      [22]    tilt (sqrt(roll^2+pitch^2))
      [23]    angular vel magnitude
      [24:29] tilt history (5)
      [29:34] angvel magnitude history (5)
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        stage: str = "hover",
        control_hz: float = 20.0,
        spawn_altitude: float = 15.0,
        teacher_mode: bool = True,
        enable_logging: bool = False
    ):
        super().__init__()

        # AirSim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Timing & config
        self.time_step = 1.0 / control_hz
        self.spawn_altitude = float(spawn_altitude)
        self.enable_logging = enable_logging

        # Modes
        self.teacher_mode = bool(teacher_mode)  # True: rates API; False: raw PWM (agent)
        self.stage = stage

        # Hover PWM & mixing for PWM mode
        self.hover_pwm = 0.63
        self.mix = 0.02  # gentle

        # Episode management
        self.current_step = 0
        self.max_steps = 500

        # Histories
        self.tilt_hist = deque(maxlen=5)
        self.angvel_hist = deque(maxlen=5)

        # Gym spaces (34D obs, 4D action)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Previous action (for obs & smoothness)
        self.prev_action = np.zeros(4, dtype=np.float32)

    # -------- Public toggles --------
    def set_teacher_mode(self, on: bool = True):
        self.teacher_mode = bool(on)

    def set_pwm_mode(self):
        self.teacher_mode = False

    def set_rates_mode(self):
        self.teacher_mode = True

    # -------- Core Gym API --------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Reset sim & re-arm
        self.client.reset()
        time.sleep(0.5)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Takeoff via PWM to avoid reliance on GPS
        self._send_pwm(0.66, 0.66, 0.66, 0.66, 2.0, join=True)

        # Let it settle briefly at hover
        self._send_pwm(0.63, 0.63, 0.63, 0.63, 1.0, join=True)

        # Move to target altitude (position mode is stable)
        self.client.moveToPositionAsync(0, 0, -self.spawn_altitude, 3).join()
        time.sleep(0.5)

        # Reset counters & histories
        self.current_step = 0
        self.prev_action[:] = 0.0
        self.tilt_hist.clear()
        self.angvel_hist.clear()
        for _ in range(5):
            self.tilt_hist.append(0.0)
            self.angvel_hist.append(0.0)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray, teacher_mode: Optional[bool] = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Returns: obs, reward, terminated, truncated, info
        """
        if teacher_mode is None:
            teacher_mode = self.teacher_mode

        # Clip incoming action
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        roll_rate, pitch_rate, yaw_rate, throttle = [to_float(v) for v in action]

        if teacher_mode:
            # Use AirSim stabilizer (rates + throttle)
            self.client.moveByAngleRatesThrottleAsync(
                float(roll_rate), float(pitch_rate), float(yaw_rate),
                float(0.5 + 0.5 * (throttle)),  # map [-1,1] -> [0,1] then trim below
                float(self.time_step)
            )
        else:
            # Raw PWM (agent). Start from hover, add gentle mixing
            base = self.hover_pwm
            mix = self.mix

            # map throttle [-1,1] -> delta around base (keep conservative)
            t_delta = 0.10 * float(throttle)

            fr = base - mix*roll_rate - mix*pitch_rate - mix*yaw_rate + t_delta
            rl = base + mix*roll_rate - mix*pitch_rate + mix*yaw_rate + t_delta
            fl = base - mix*roll_rate + mix*pitch_rate + mix*yaw_rate + t_delta
            rr = base + mix*roll_rate + mix*pitch_rate - mix*yaw_rate + t_delta

            fr, rl, fl, rr = [float(v) for v in clamp01(np.array([fr, rl, fl, rr]))]
            # keep minimum lift margin
            fr = max(fr, 0.58); rl = max(rl, 0.58); fl = max(fl, 0.58); rr = max(rr, 0.58)

            self._send_pwm(fr, rl, fl, rr, self.time_step, join=False)

        time.sleep(self.time_step)

        # Build obs, compute reward/termination
        obs = self._get_obs()
        reward = self._reward(obs)
        terminated, truncated, info = self._termination(obs)

        # Track
        self.prev_action = action
        self.current_step += 1

        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            self.client.reset()
        except Exception:
            pass

    # -------- Internals --------
    def _send_pwm(self, fr, rl, fl, rr, duration, join=False):
        h = self.client.moveByMotorPWMsAsync(float(fr), float(rl), float(fl), float(rr), float(duration))
        if join:
            h.join()

    def _get_obs(self) -> np.ndarray:
        state = self.client.getMultirotorState().kinematics_estimated

        vel = state.linear_velocity
        ang = state.angular_velocity
        ori = state.orientation
        acc = state.linear_acceleration
        pos = state.position

        roll, pitch, yaw = airsim.to_eularian_angles(ori)
        altitude = -pos.z_val
        vert_vel = -vel.z_val

        tilt = float(np.sqrt(roll**2 + pitch**2))
        ang_mag = float(np.linalg.norm([ang.x_val, ang.y_val, ang.z_val]))

        self.tilt_hist.append(tilt)
        self.angvel_hist.append(ang_mag)

        obs = np.array([
            vel.x_val, vel.y_val, vel.z_val,                         # 0..2
            ang.x_val, ang.y_val, ang.z_val,                         # 3..5
            ori.w_val, ori.x_val, ori.y_val, ori.z_val,             # 6..9
            acc.x_val, acc.y_val, acc.z_val,                        # 10..12
            roll, pitch, yaw,                                       # 13..15
            altitude,                                               # 16
            vert_vel,                                               # 17
            *self.prev_action.tolist(),                             # 18..21
            tilt,                                                   # 22
            ang_mag,                                                # 23
            *list(self.tilt_hist),                                  # 24..28 (5)
            *list(self.angvel_hist)                                 # 29..33 (5)
        ], dtype=np.float32)

        # Sanity: always 34
        # len = 34
        return obs

    def _reward(self, obs: np.ndarray) -> float:
        roll, pitch = float(obs[13]), float(obs[14])
        tilt = float(np.sqrt(roll**2 + pitch**2))
        ang_mag = float(obs[23])
        altitude = float(obs[16])
        vert_vel = float(obs[17])

        # Penalize attitude error + angular rate
        rew = 0.0
        rew -= 8.0 * (tilt ** 2)
        rew -= 2.0 * (ang_mag ** 2)

        # Altitude tracking to spawn_altitude
        rew -= 0.5 * (abs(altitude - self.spawn_altitude))

        # Penalize fast downward velocity near ground
        if altitude < 3.0 and vert_vel < -0.8:
            rew -= 3.0 * abs(vert_vel + 0.8)

        # Small survival bonus
        if altitude > 1.0:
            rew += 0.25

        return rew

    def _termination(self, obs: np.ndarray) -> Tuple[bool, bool, Dict]:
        roll, pitch = float(obs[13]), float(obs[14])
        tilt = float(np.sqrt(roll**2 + pitch**2))
        altitude = float(obs[16])

        info: Dict = {}
        terminated = False
        truncated = False

        # Hard failure: ground / inverted
        if altitude <= 0.5:
            truncated = True
            info["termination_reason"] = "ground"
        elif tilt > np.pi * 0.9:
            truncated = True
            info["termination_reason"] = "inverted"
        elif self.current_step >= self.max_steps:
            truncated = True
            info["termination_reason"] = "time_limit"

        # Collision info (if available)
        try:
            col = self.client.simGetCollisionInfo()
            if getattr(col, "has_collided", False):
                truncated = True
                info["termination_reason"] = "collision"
        except Exception:
            pass

        return terminated, truncated, info


# ---------------------------
# Quick smoke test
# ---------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Smoke Test: Teacher rates for 5 seconds")
    print("=" * 70)

    env = PostImpactRecoveryEnvRate(teacher_mode=True, spawn_altitude=15.0, control_hz=20.0)
    teacher = TeacherPD(alt_target=15.0)

    obs, _ = env.reset()

    total = 0.0
    for t in range(100):  # ~5 seconds @ 20 Hz
        act = teacher.get_action(obs)           # rates+throttle
        # Map teacher outputs (already in proper ranges) to env action space [-1,1] where appropriate.
        # Here we pass them directly; step() will use teacher_mode=True rates API.
        obs, r, term, trunc, info = env.step(act, teacher_mode=True)
        total += r
        if (t + 1) % 20 == 0:
            alt = float(obs[16]); tilt = float(np.degrees(np.sqrt(obs[13]**2 + obs[14]**2)))
            print(f"t={(t+1)/20:.1f}s  alt={alt:5.2f} m  tilt={tilt:5.1f}°  r={r:6.2f}")
        if term or trunc:
            print("Ended:", info.get("termination_reason"))
            break

    print(f"\nTotal reward ~ {total:.2f}")
    env.close()
