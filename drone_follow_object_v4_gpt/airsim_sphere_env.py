"""
AirSimSphereChaseEnv_v2
------------------------------------------------
A robust Gymnasium environment for a drone in AirSim
to chase and "hit" a floating sphere while maintaining
the same altitude as the target.

Key Features:
  ✅ Correct NED frame (Z positive = down)
  ✅ PPO-friendly continuous actions [vx, vy, vz]
  ✅ Altitude lock-on: rewards matching target altitude
  ✅ Smooth dense rewards (distance, alignment, altitude)
  ✅ Gravity compensation bias
  ✅ Fully msgpack-safe (no numpy floats passed to AirSim)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
import airsim
from typing import Optional, Dict, Any


class AirSimSphereChaseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        ip: str = "127.0.0.1",
        start_pos=(0.0, 0.0, -10.0),
        max_speed: float = 6.0,
        dt: float = 0.05,
        hit_threshold: float = 2.0,
        altitude_band: float = 1.0,
        far_fail_distance: float = 120.0,
        max_episode_steps: int = 1000,
        ring_r_min: float = 10.0,
        ring_r_max: float = 30.0,
        z_min: float = -25.0,
        z_max: float = -8.0,
        visualize: bool = True,
        persistent_markers: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.ip = ip
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.max_speed = float(max_speed)
        self.dt = float(dt)
        self.hit_threshold = float(hit_threshold)
        self.altitude_band = float(altitude_band)
        self.far_fail_distance = float(far_fail_distance)
        self.max_episode_steps = int(max_episode_steps)
        self.ring_r_min = float(ring_r_min)
        self.ring_r_max = float(ring_r_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.visualize = bool(visualize)
        self.persistent_markers = bool(persistent_markers)
        self.gravity_bias = 0.3  # upward bias to fight gravity

        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Observation: [dx, dy, dz, distance, vx, vy, vz]
        high = np.array([200, 200, 200, 300, 20, 20, 20], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Action: continuous velocities [vx, vy, vz]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Connect to AirSim
        self.client = airsim.MultirotorClient(ip=self.ip)
        self.client.confirmConnection()
        self._ensure_ready()

        self.steps = 0
        self.hits = 0
        self.prev_dist = 0.0
        self.target = np.zeros(3, dtype=np.float32)

    # ---------- Utility ----------
    def _ensure_ready(self):
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def _pf(self, x):
        if isinstance(x, np.generic):
            return float(x.item())
        return float(x)

    def _create_sphere(self, pos, radius=2.0, color=[1.0, 0.0, 0.0, 1.0]):
        """Visualize target sphere"""
        if not self.visualize:
            return
        points = []
        num_points = 150
        phi = math.pi * (3.0 - math.sqrt(5.0))
        px, py, pz = map(self._pf, pos)
        r = self._pf(radius)
        for i in range(num_points):
            y_off = 1 - (i / float(num_points - 1)) * 2
            r_at_y = math.sqrt(max(0.0, 1 - y_off * y_off))
            theta = phi * i
            x = math.cos(theta) * r_at_y * r
            y = math.sin(theta) * r_at_y * r
            z = y_off * r
            points.append(airsim.Vector3r(px + x, py + y, pz + z))
        self.client.simPlotPoints(points, color_rgba=color, size=15.0, duration=self.dt * 10000, is_persistent=self.persistent_markers)

    def _clear_markers(self):
        if self.visualize:
            self.client.simFlushPersistentMarkers()

    def _get_drone_state(self):
        s = self.client.getMultirotorState()
        p = s.kinematics_estimated.position
        v = s.kinematics_estimated.linear_velocity
        pos = np.array([p.x_val, p.y_val, p.z_val], dtype=np.float32)
        vel = np.array([v.x_val, v.y_val, v.z_val], dtype=np.float32)
        return pos, vel

    def _spawn_target(self):
        """Random sphere spawn"""
        angle = self.np_random.uniform(0, 2 * math.pi)
        r = self.np_random.uniform(self.ring_r_min, self.ring_r_max)
        tx = r * math.cos(angle)
        ty = r * math.sin(angle)
        tz = self.np_random.uniform(self.z_min, self.z_max)
        self.target = np.array([tx, ty, tz], dtype=np.float32)
        self._clear_markers()
        self._create_sphere(self.target, radius=2.0, color=[1.0, 0.0, 0.0, 1.0])

    def _obs(self, pos, vel):
        rel = self.target - pos
        dist = float(np.linalg.norm(rel))
        obs = np.array([rel[0], rel[1], rel[2], dist, vel[0], vel[1], vel[2]], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self._ensure_ready()
        self.client.takeoffAsync().join()

        noise = self.np_random.uniform(-2, 2, size=3)
        start = self.start_pos + noise
        self.client.moveToPositionAsync(float(start[0]), float(start[1]), float(start[2]), 3).join()

        self._spawn_target()
        pos, vel = self._get_drone_state()
        self.prev_dist = float(np.linalg.norm(self.target - pos))
        self.steps = 0
        self.hits = 0

        obs = self._obs(pos, vel)
        info = {"hits": self.hits, "target": self.target.copy()}
        return obs, info

    def step(self, action):
        self.steps += 1
        pos, vel = self._get_drone_state()

        # Scale action to velocities (m/s)
        scaled = (action * self.max_speed).astype(np.float32)
        vx, vy, vz = map(self._pf, scaled)

        # FIX: invert Z (NED frame)
        vz = -vz + self.gravity_bias

        # Clamp altitude: prevent ground collision
        if pos[2] > -2.0:
            vz = min(vz, 0.0)

        # Send velocity command
        self.client.moveByVelocityAsync(
            self._pf(vx), self._pf(vy), self._pf(vz), self._pf(self.dt),
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=self._pf(0.0))
        )
        time.sleep(self.dt)

        pos, vel = self._get_drone_state()
        rel = self.target - pos
        dist = float(np.linalg.norm(rel))
        dz = float(rel[2])

        # ---------- REWARD SHAPING ----------
        reward = 0.0
        terminated = False
        truncated = False

        # 1. Distance-based shaping
        progress = self.prev_dist - dist
        reward += 2.0 * progress
        reward += 5.0 / (dist + 1.0)
        reward -= 0.01 * dist

        # 2. Altitude matching
        alt_diff = abs(dz)
        if alt_diff < self.altitude_band:
            reward += 2.0
        else:
            reward -= 0.5 * alt_diff

        # 3. Directional alignment
        rel_dir = rel / (np.linalg.norm(rel) + 1e-6)
        alignment = np.dot(vel, rel_dir)
        reward += 0.2 * alignment

        # 4. Idle penalty
        if dist > self.hit_threshold + 2.0 and np.linalg.norm(action) < 0.1:
            reward -= 0.3

        # 5. Hit condition
        if dist <= self.hit_threshold:
            reward += 100.0
            self.hits += 1
            self.client.hoverAsync().join()
            time.sleep(0.2)
            self._spawn_target()

        # 6. Too far away
        if dist > self.far_fail_distance:
            reward -= 50.0
            terminated = True

        # 7. Episode timeout
        if self.steps >= self.max_episode_steps:
            truncated = True

        self.prev_dist = dist
        obs = self._obs(pos, vel)
        info = {"hits": self.hits, "distance": dist, "target": self.target.copy()}
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self.client.hoverAsync().join()
            self.client.landAsync().join()
        except Exception:
            pass
        self.client.armDisarm(False)
        self.client.enableApiControl(False)


# ---------- Manual Test ----------
if __name__ == "__main__":
    env = AirSimSphereChaseEnv(visualize=True)
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(300):
        # Try slight forward + upward movement
        action = np.array([0.2, 0.8, 0.3], dtype=np.float32)
        obs, r, done, trunc, info = env.step(action)
        total_reward += r
        if done or trunc:
            break
    print("Total reward:", total_reward, "Hits:", info.get("hits", 0))
    env.close()
