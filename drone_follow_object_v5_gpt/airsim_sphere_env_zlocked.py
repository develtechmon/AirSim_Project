"""
Drone Sphere Chasing Environment for AirSim
Continuous action space + PPO-ready + msgpack-safe
Fixed version by Lukas & ChatGPT

Key improvements:
✅ Fixes msgpack numpy.float32 serialization error
✅ Adds solid termination & truncation (too far, altitude, inactivity)
✅ Adds pf() wrapper for all AirSim float arguments
✅ Keeps visual target sphere for debugging
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import time


# -------------- Helper to prevent msgpack errors --------------
def pf(x):
    """Convert NumPy scalars or arrays to plain Python float."""
    return float(x.item() if isinstance(x, np.generic) else x)


class DroneChaseEnv(gym.Env):
    """
    PPO-ready environment for training a drone to chase and hit a moving sphere in AirSim.
    """

    def __init__(self,
                 max_altitude=50,
                 min_altitude=-50,
                 hit_threshold=2.0,
                 max_distance=100,
                 spawn_radius=30):
        super().__init__()

        # ---------------- Environment Config ----------------
        self.max_altitude = max_altitude
        self.min_altitude = min_altitude
        self.hit_threshold = hit_threshold
        self.max_distance = max_distance
        self.spawn_radius = spawn_radius

        # Continuous Action Space: [vx, vy, vz]
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0], dtype=np.float32)
        )

        # Observation: [dx, dy, dz, vx, vy, vz, distance, altitude_diff]
        self.observation_space = spaces.Box(
            low=np.array([-200, -200, -200, -10, -10, -10, 0, -100]),
            high=np.array([200, 200, 200, 10, 10, 10, 200, 100]),
            dtype=np.float32
        )

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # Tracking Variables
        self.previous_distance = None
        self.target_position = None
        self.previous_position = None
        self.episode_steps = 0
        self.max_episode_steps = 500
        self.steps_without_progress = 0
        self.max_steps_without_progress = 25

        print("✅ Drone Chase Environment Initialized")

    # ==========================================================
    # RESET
    # ==========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("\n🔄 Resetting environment...")

        # Reset AirSim
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        time.sleep(0.5)
        self.client.moveToPositionAsync(pf(0), pf(0), pf(-10), pf(5)).join()

        # Clear visual markers
        self.client.simFlushPersistentMarkers()

        # Spawn target
        self._spawn_target()

        # Initialize state
        drone_pos = self._get_position()
        self.previous_distance = self._distance(drone_pos, self.target_position)
        self.previous_position = drone_pos.copy()
        self.episode_steps = 0
        self.steps_without_progress = 0

        obs = self._get_observation()
        info = {"episode_steps": self.episode_steps}
        return obs, info

    # ==========================================================
    # STEP
    # ==========================================================
    def step(self, action):
        self.episode_steps += 1

        # --- Apply action safely ---
        vx, vy, vz = [pf(a) for a in np.clip(action, -5, 5)]
        self.client.moveByVelocityAsync(
            vx, vy, vz,
            duration=pf(1.0),
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, pf(0))
        ).join()

        # --- Get current state ---
        drone_pos = self._get_position()
        drone_vel = self._get_velocity()
        current_distance = self._distance(drone_pos, self.target_position)
        altitude_diff = abs(drone_pos[2] - self.target_position[2])

        # --- Movement detection ---
        movement = self._distance(drone_pos, self.previous_position)
        if movement < 0.5:
            self.steps_without_progress += 1
        else:
            self.steps_without_progress = 0
        self.previous_position = drone_pos.copy()

        # --- Termination Conditions ---
        hit_target = current_distance <= self.hit_threshold
        too_far = current_distance > self.max_distance
        altitude_violation = altitude_diff > 20.0
        inactive = self.steps_without_progress >= self.max_steps_without_progress

        # --- Reward Calculation ---
        reward = self._calculate_reward(
            current_distance=current_distance,
            previous_distance=self.previous_distance,
            altitude_diff=altitude_diff,
            hit_target=hit_target,
            out_of_bounds=too_far,
            altitude_violation=altitude_violation,
            action_magnitude=np.linalg.norm(action),
            inactive=inactive
        )

        self.previous_distance = current_distance

        # --- Episode End Conditions ---
        terminated = bool(hit_target)
        truncated = bool(
            too_far or altitude_violation or inactive or self.episode_steps >= self.max_episode_steps
        )

        # --- Respawn Target on Hit ---
        if hit_target:
            print(f"🎯 Target HIT at step {self.episode_steps}! Respawning...")
            self._spawn_target()
            self.steps_without_progress = 0
            self.previous_position = drone_pos.copy()
            self.previous_distance = self._distance(drone_pos, self.target_position)

        obs = self._get_observation()
        info = {
            "steps": self.episode_steps,
            "distance": current_distance,
            "altitude_diff": altitude_diff,
            "inactive": inactive,
            "too_far": too_far,
            "hit_target": hit_target
        }

        return obs, reward, terminated, truncated, info

    # ==========================================================
    # REWARD FUNCTION
    # ==========================================================
    def _calculate_reward(self, current_distance, previous_distance, altitude_diff,
                          hit_target, out_of_bounds, altitude_violation,
                          action_magnitude, inactive):
        reward = 0.0

        # Success
        if hit_target:
            return 500.0

        # Out of bounds
        if out_of_bounds:
            return -200.0

        # Altitude too far
        if altitude_violation:
            return -150.0

        # Inactive drone penalty
        if inactive:
            reward -= 100.0

        # Encourage movement
        if action_magnitude > 1.0:
            reward += action_magnitude * 2.0
        elif action_magnitude < 0.3:
            reward -= 5.0

        # Distance shaping
        delta = previous_distance - current_distance
        if delta > 0:
            reward += delta * 20.0
        else:
            reward -= abs(delta) * 10.0

        # Altitude shaping
        if altitude_diff < 3.0:
            reward += 5.0
        else:
            reward -= altitude_diff * 1.5

        # Small time penalty
        reward -= 1.0

        return float(reward)

    # ==========================================================
    # UTILITIES
    # ==========================================================
    def _get_position(self):
        p = self.client.getMultirotorState().kinematics_estimated.position
        return np.array([pf(p.x_val), pf(p.y_val), pf(p.z_val)], dtype=np.float32)

    def _get_velocity(self):
        v = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        return np.array([pf(v.x_val), pf(v.y_val), pf(v.z_val)], dtype=np.float32)

    def _distance(self, a, b):
        return float(np.linalg.norm(a - b))

    def _spawn_target(self):
        """Spawn target sphere visually in AirSim."""
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(10, self.spawn_radius)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = np.random.uniform(-30, -10)
        self.target_position = np.array([pf(x), pf(y), pf(z)], dtype=np.float32)

        self.client.simFlushPersistentMarkers()
        self._create_point_cloud_sphere(self.target_position)
        print(f"🎯 Target spawned at ({x:.1f}, {y:.1f}, {z:.1f})")

    def _create_point_cloud_sphere(self, position, radius=2.0, color=[1.0, 0.0, 0.0, 1.0]):
        """Draw a red sphere at the target location."""
        points = []
        num_points = 150
        phi = math.pi * (3. - math.sqrt(5.))
        for i in range(num_points):
            y_off = 1 - (i / float(num_points - 1)) * 2
            r_y = math.sqrt(1 - y_off * y_off)
            th = phi * i
            x = math.cos(th) * r_y * radius
            y = math.sin(th) * r_y * radius
            z = y_off * radius
            point = airsim.Vector3r(
                pf(position[0] + x),
                pf(position[1] + y),
                pf(position[2] + z)
            )
            points.append(point)
        self.client.simPlotPoints(
            points,
            color_rgba=color,
            size=pf(15.0),
            duration=pf(120.0),
            is_persistent=True
        )

    def _get_observation(self):
        pos = self._get_position()
        vel = self._get_velocity()
        dx, dy, dz = self.target_position - pos
        distance = self._distance(pos, self.target_position)
        altitude_diff = dz
        obs = np.array([dx, dy, dz, vel[0], vel[1], vel[2], distance, altitude_diff], dtype=np.float32)
        return obs

    def close(self):
        """Release control and stop AirSim."""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("✅ Environment closed safely.")
