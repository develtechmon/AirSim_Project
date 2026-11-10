"""
PERFORMANCE-GATED CURRICULUM ENVIRONMENT (ArduPilot SITL + AirSim)
==================================================================
Same job, same class name, same curriculum and same info keys as your AirSim
drone_flip_recovery_env_injector_gated.py. Only moves to the next difficulty level
when the current level is MASTERED:

    Level 0 EASY   intensity 0.7-0.9x  -> needs 80% recovery (last 50, min 50 episodes at level)
    Level 1 MEDIUM intensity 0.9-1.1x  -> needs 70%
    Level 2 HARD   intensity 1.1-1.5x  -> target 60%

What is the same as AirSim:
    13 observations (position, velocity, quaternion, body rates), 3 actions = velocity N/E/D (m/s)
    -> the Stage 2 network is the starting point
    hover at 30 m, start within +/-1 m, 600 steps at 20 Hz (30 s), disturbance at step 20-50
    disturbances: bird_attack, flip, spin; wind gusts (AirSim simSetWind) as Stage 2
    recovered = |body rate| < 1.2 rad/s AND upright (up_z > 0.7) AND height > 3 m AND h-speed < 3 m/s

What is different (ArduPilot):
    - the impact is made by ArduPilot rate commands (disturbance_injector.py, checked with impact_check.py)
    - the drone is NOT teleported between episodes: it flies back to the start (no client.reset())
    - VIRTUAL GROUND 5 m: an episode ends as 'crash' when the drone could no longer stop above
      5 m (height minus braking distance) - the real floor is never reached, SITL never restarts.
      ArduPilot then takes over at once (reset flies it back up to 30 m).
    - observation scaling: Stage 1's obs_norm.npz (same as Stage 2), not VecNormalize,
      so the Stage 2 network fits exactly
    - reward: small numbers like Stage 2 (+1 per step alive, minus spin / tilt / falling / distance,
      +50 once for recovering (more for faster), -100 crash). Your AirSim reward (up to thousands
      per step) gave value losses around 1e6, and paid MORE while still recovering than after.
    - a tumble is counted for the curriculum ONCE per episode, at the end (AirSim appended a 1
      every step after recovery, which inflated the rate). A recovery followed by a crash = failed.
    - safety pilot: while PPO updates its network (several seconds, no commands), a background
      thread turns the wind off and holds the drone; that episode is cut ('update_pause') and
      NOT counted (not in the curriculum, not in the CSV).
    - a REAL crash (disarmed in the air / on the floor) raises an error: training saves and stops.
"""

import threading
import time
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ap_link import APLink, RateKeeper
from disturbance_injector import DisturbanceInjector, DisturbanceType
from pid_expert import PIDExpertHover

HZ = 20.0
FLOOR = 5.0                # m virtual ground
BRAKE = 5.0                # m/s^2 braking assumed for the virtual-ground check (conservative)
CLIP_OBS = 10.0
ACT_LIMIT = 5.0            # same action space as Stage 1/2 (and AirSim Stage 3)
VMAX_XY_SEND = 5.0         # m/s sent to ArduPilot (AirSim Stage 3 allowed +/-5)
VMAX_Z_SEND = 3.0          # m/s
FLY_GAIN, FLY_XY, FLY_Z = 0.8, 2.0, 2.0     # flying back to the start between episodes
LEVEL_NAMES = ["EASY", "MEDIUM", "HARD"]
LEVEL_RANGES = [(0.7, 0.9), (0.9, 1.1), (1.1, 1.5)]
THRESHOLDS = {0: 0.80, 1: 0.70, 2: 0.60}
TYPES = [DisturbanceType.BIRD_ATTACK, DisturbanceType.FLIP, DisturbanceType.SPIN]


class WindSchedule:
    """Same gust model as Stage 2: random direction, 0..max, new gust every 1-3 s."""

    def __init__(self, max_wind, rng):
        self.max_wind, self.rng = max_wind, rng
        self.timer, self.interval = 0.0, rng.uniform(1.0, 3.0)
        self.current = self._new()

    def _new(self):
        if self.max_wind <= 0:
            return np.zeros(3)
        theta, phi = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(0, np.pi)
        s = self.rng.uniform(0, self.max_wind)
        return np.array([s * np.sin(phi) * np.cos(theta), s * np.sin(phi) * np.sin(theta), s * np.cos(phi) * 0.3])

    def step(self, dt):
        self.timer += dt
        if self.timer >= self.interval:
            self.current = self._new()
            self.timer, self.interval = 0.0, self.rng.uniform(1.0, 3.0)
            return True
        return False


class DroneFlipRecoveryEnvGated(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_norm_path, target_altitude=30.0, max_steps=600, wind_strength=5.0,
                 flip_prob=1.0, seed=None, debug=False, connect="tcp:127.0.0.1:5762",
                 curriculum_frozen=False, start_level=0):
        super().__init__()
        norm = np.load(obs_norm_path)
        self.mean, self.std = norm["mean"], norm["std"]
        self.target_altitude, self.max_steps = target_altitude, max_steps
        self.wind_strength, self.disturbance_prob, self.debug = wind_strength, flip_prob, debug
        self.action_space = spaces.Box(-ACT_LIMIT, ACT_LIMIT, (3,), np.float32)
        self.observation_space = spaces.Box(-CLIP_OBS, CLIP_OBS, (13,), np.float32)
        self.rng = np.random.RandomState(seed)

        import airsim
        self._airsim = airsim
        self.sim = airsim.MultirotorClient()
        self.sim.confirmConnection()
        self.link = APLink(connect, rate_hz=50)
        self.link.wait_state()
        self.disturbance_injector = DisturbanceInjector(self.link, self.rng, floor=FLOOR)
        self.teacher = PIDExpertHover()
        self.home = None
        self.target = None

        # episode tracking
        self.episode_steps = 0
        self.episode_count = 0
        self.stable_steps = 0
        self.wind = WindSchedule(0.0, self.rng)
        self.rk = RateKeeper(HZ)
        self._reset_tracking()

        # performance-gated curriculum state (same as AirSim)
        self.curriculum_level = start_level
        self.curriculum_frozen = curriculum_frozen       # tests: never advance
        self.level_names = LEVEL_NAMES
        self.recent_recoveries = deque(maxlen=50)
        self.advancement_thresholds = THRESHOLDS
        self.min_episodes_per_level = 50
        self.episodes_at_current_level = 0
        self.level_advanced = False
        self.advancement_info = {}

        # safety pilot (PPO update pauses)
        self._last_step_t = time.time()
        self._active = False
        self._injecting = False
        self._took_over = False
        self._stop = False
        self._thread = threading.Thread(target=self._safety_pilot, daemon=True)
        self._thread.start()
        if self.debug:
            print("✓ PERFORMANCE-GATED Curriculum Environment (ArduPilot)")
            print(f"  - Starting level: {self.level_names[self.curriculum_level]}")
            print(f"  - Altitude: {self.target_altitude}m | virtual ground {FLOOR}m")

    # ------------------------------------------------------------ helpers
    def _reset_tracking(self):
        self.disturbance_initiated = False
        self.disturbance_recovered = False
        self.disturbance_start_step = 0
        self.recovery_steps = 0
        self.disturbance_info = {}
        self.previous_ang_vel_magnitude = 0.0
        self.ep = {"min_alt": 1e9, "max_rate": 0.0, "ret": 0.0}

    def _set_wind(self, w):
        try:
            self.sim.simSetWind(self._airsim.Vector3r(float(w[0]), float(w[1]), float(w[2])))
        except Exception:
            pass

    def _norm(self, o):
        return np.clip((o - self.mean) / self.std, -CLIP_OBS, CLIP_OBS).astype(np.float32)

    def _get_observation(self):
        return self._norm(self.link.obs(self.target))

    def _up_z(self):
        w, x, y, z = self.link.q
        return 1.0 - 2.0 * (x * x + y * y)

    def _brake_floor(self):
        v_down = max(0.0, float(self.link.vel[2]))
        return self.link.alt - v_down ** 2 / (2 * BRAKE)

    def _check_real_crash(self):
        if not self.link.armed or self.link.alt < 0.3:
            raise RuntimeError("💥 REAL CRASH (disarmed or on the floor) - restart SITL, then --resume")

    def _fly_to(self, goal, hold_s, max_s=90.0):
        """ArduPilot flies to goal with gentle velocity commands, then holds calmly for hold_s."""
        rk = RateKeeper(HZ)
        t0, calm_since = time.time(), None
        while time.time() - t0 < max_s:
            self.link.update()
            self._check_real_crash()
            err = goal - self.link.pos
            v = np.clip(FLY_GAIN * err, -FLY_XY, FLY_XY)
            self.link.send_velocity(v[0], v[1], float(np.clip(v[2], -FLY_Z, FLY_Z)))
            rk.wait()
            calm = (np.linalg.norm(err) < 0.3 and np.linalg.norm(self.link.vel) < 0.3
                    and self.link.tilt_deg < 5)
            calm_since = (calm_since or time.time()) if calm else None
            if calm_since and time.time() - calm_since >= hold_s:
                return True
        print("   (did not reach the start point calmly - continuing)")
        return False

    def _safety_pilot(self):
        """Holds the drone (wind off) whenever no step came for > 0.15 s (PPO update)."""
        sim = None
        while not self._stop:
            time.sleep(0.05)
            if not self._active or self._injecting or time.time() - self._last_step_t < 0.15:
                continue
            if not self._took_over:
                self._took_over = True
                try:
                    if sim is None:
                        sim = self._airsim.MultirotorClient()
                    sim.simSetWind(self._airsim.Vector3r(0, 0, 0))
                except Exception:
                    pass
            self.link.update()
            o = self.link.obs(self.target)
            self.link.send_velocity(*self.teacher.get_action({"position": o[0:3], "velocity": o[3:6]}))

    def _get_intensity_for_level(self):
        lo, hi = LEVEL_RANGES[self.curriculum_level]
        return self.rng.uniform(lo, hi)

    def _check_curriculum_advancement(self):
        """Same rule as AirSim: min 50 episodes at level, >= 30 counted, rate >= threshold."""
        if self.curriculum_frozen or self.curriculum_level >= 2:
            return
        if self.episodes_at_current_level < self.min_episodes_per_level or len(self.recent_recoveries) < 30:
            return
        rate = float(np.mean(list(self.recent_recoveries)))
        threshold = self.advancement_thresholds[self.curriculum_level]
        if rate >= threshold:
            old = self.curriculum_level
            self.curriculum_level += 1
            self.episodes_at_current_level = 0
            self.recent_recoveries.clear()
            self.level_advanced = True
            self.advancement_info = {"old_level": old, "new_level": self.curriculum_level,
                                     "recovery_rate": rate, "threshold": threshold,
                                     "episode": self.episode_count}
            print("\n" + "=" * 70)
            print("🎓 CURRICULUM ADVANCEMENT!")
            print("=" * 70)
            print(f"   Level {old} ({self.level_names[old]}) MASTERED!")
            print(f"   Recovery rate: {rate * 100:.1f}% (needed {threshold * 100:.0f}%)")
            print(f"   Advancing to Level {self.curriculum_level} ({self.level_names[self.curriculum_level]})")
            print("   💾 MODEL WILL BE AUTO-SAVED!")
            print("=" * 70 + "\n")

    def set_curriculum_state(self, level, episodes_at_level, recent):
        """Used by --resume: rebuild the curriculum from the episode log."""
        self.curriculum_level = int(level)
        self.episodes_at_current_level = int(episodes_at_level)
        self.recent_recoveries.clear()
        self.recent_recoveries.extend(int(x) for x in recent[-50:])

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "seed" in options:
            self.rng.seed(options["seed"])
        self._active = False
        self._took_over = False
        self._set_wind(np.zeros(3))
        self.link.update()
        if not self.link.armed and self.home is None:      # first episode: take off
            self.link.takeoff_nogps(self.target_altitude)
        self._check_real_crash()
        if self.home is None:
            self.link.update()
            self.home = self.link.pos.copy()
        self.target = np.array([self.home[0], self.home[1], -self.target_altitude])
        start = self.target + np.array([self.rng.uniform(-1, 1), self.rng.uniform(-1, 1), 0.0])
        self._fly_to(start, hold_s=1.0)

        self.wind = WindSchedule(self.wind_strength, self.rng)
        self._set_wind(self.wind.current)
        self.episode_steps = 0
        self.stable_steps = 0
        self._reset_tracking()
        self.will_have_disturbance = self.rng.rand() < self.disturbance_prob
        if self.will_have_disturbance:
            self.disturbance_trigger_step = int(self.rng.randint(20, 50))
            self.disturbance_type = TYPES[self.rng.randint(len(TYPES))]
            if self.debug:
                print(f"   ⚠️  Disturbance scheduled: {self.disturbance_type.value} at step "
                      f"{self.disturbance_trigger_step} | Level {self.curriculum_level} "
                      f"({self.level_names[self.curriculum_level]}) | episodes at level "
                      f"{self.episodes_at_current_level}")
        else:
            self.disturbance_trigger_step = -1
            if self.debug:
                print(f"   ✅ No disturbance this episode | Level {self.curriculum_level}")
        self.episode_count += 1
        self.episodes_at_current_level += 1
        self.rk = RateKeeper(HZ)
        self.link.update()
        obs = self._get_observation()
        self._last_step_t = time.time()
        self._active = True
        return obs, {}

    def step(self, action):
        if self._took_over:                               # PPO paused: cut this episode (not counted)
            self._took_over = False
            self._last_step_t = time.time()
            self.link.update()
            info = self._episode_info("update_pause")
            return self._get_observation(), 0.0, False, True, info

        self.episode_steps += 1
        # ---- the impact (blocking, like the AirSim .join())
        if self.will_have_disturbance and not self.disturbance_initiated \
                and self.episode_steps >= self.disturbance_trigger_step:
            intensity = self._get_intensity_for_level()
            self._injecting = True
            try:
                self.disturbance_info = self.disturbance_injector.inject_disturbance(self.disturbance_type, intensity)
            finally:
                self._injecting = False
                self._last_step_t = time.time()
            self.disturbance_initiated = True
            self.disturbance_start_step = self.episode_steps
            if self.debug:
                d = self.disturbance_info
                print(f"   🐦 DISTURBANCE: {d['type']} | intensity {intensity:.2f} | "
                      f"turns {d['turns']:.1f}/{d['turns_planned']:.1f} | peak {d['peak_rate']:.1f} rad/s"
                      + (" | cut early" if d["cut"] else ""))

        # ---- the network's action (velocity N/E/D, same as Stage 2)
        a = np.clip(np.asarray(action, dtype=np.float64), -ACT_LIMIT, ACT_LIMIT)
        self.link.send_velocity(float(np.clip(a[0], -VMAX_XY_SEND, VMAX_XY_SEND)),
                                float(np.clip(a[1], -VMAX_XY_SEND, VMAX_XY_SEND)),
                                float(np.clip(a[2], -VMAX_Z_SEND, VMAX_Z_SEND)))
        if self.wind.step(1.0 / HZ):
            self._set_wind(self.wind.current)
        self.rk.wait()
        self._last_step_t = time.time()
        self.link.update()
        if not self.link.armed:
            raise RuntimeError("💥 REAL CRASH: the FC disarmed in the air - restart SITL, then --resume")

        raw = self.link.obs(self.target)
        alt = self.link.alt
        dist_from_center = float(np.linalg.norm(raw[0:2]))
        dist_from_target_alt = abs(alt - self.target_altitude)
        up_z = self._up_z()
        is_upright = up_z > 0.7
        ang_vel_magnitude = float(np.linalg.norm(self.link.rates))
        horizontal_vel = float(np.linalg.norm(self.link.vel[0:2]))
        v_down = max(0.0, float(self.link.vel[2]))
        self.ep["min_alt"] = min(self.ep["min_alt"], alt)
        self.ep["max_rate"] = max(self.ep["max_rate"], ang_vel_magnitude)

        # ---- recovery check (your AirSim criterion)
        just_recovered = False
        if self.disturbance_initiated and not self.disturbance_recovered:
            if ang_vel_magnitude < 1.2 and is_upright and alt > 3.0 and horizontal_vel < 3.0:
                self.disturbance_recovered = True
                just_recovered = True
                self.recovery_steps = self.episode_steps - self.disturbance_start_step
                if self.debug:
                    print(f"   ✅ RECOVERED! Took {self.recovery_steps} steps ({self.recovery_steps / HZ:.1f}s) | "
                          f"lowest {self.ep['min_alt']:.1f} m")

        # ---- reward (Stage 2 style: small numbers)
        reward = (1.0 - 0.05 * min(ang_vel_magnitude, 15.0) - 0.25 * (1.0 - up_z)
                  - 0.05 * min(v_down, 20.0) - 0.02 * min(dist_from_center, 15.0)
                  - 0.02 * min(dist_from_target_alt, 15.0) - 0.01 * float(np.sum(a ** 2)))
        if just_recovered:
            reward += 50.0 * float(np.clip(1.0 - (self.recovery_steps / HZ) / 10.0, 0.2, 1.0))
        if self.disturbance_recovered or not self.will_have_disturbance:
            self.stable_steps = self.stable_steps + 1 if is_upright else 0
        self.previous_ang_vel_magnitude = ang_vel_magnitude

        # ---- termination
        done, reason = False, None
        if alt < FLOOR or self._brake_floor() < FLOOR:
            reward -= 100.0
            done, reason = True, "crash"                  # virtual ground
            self.link.send_velocity(0.0, 0.0, 0.0)        # ArduPilot takes over at once
        elif alt > self.target_altitude + 10.0:
            reward -= 50.0
            done, reason = True, "too_high"
        elif dist_from_center > 15.0:
            reward -= 50.0
            done, reason = True, "too_far"
        elif self.episode_steps >= self.max_steps:
            done, reason = True, "timeout"
        self.ep["ret"] += reward

        info = {"altitude": alt, "distance": dist_from_center, "is_upright": bool(is_upright),
                "is_tumbling": ang_vel_magnitude > 1.0, "angular_velocity_mag": ang_vel_magnitude,
                "tumble_initiated": self.disturbance_initiated, "tumble_recovered": self.disturbance_recovered,
                "wind_magnitude": float(np.linalg.norm(self.wind.current)),
                "curriculum_level": self.curriculum_level, "reason": reason or "running"}
        if self.disturbance_recovered:
            info["recovery_steps"] = self.recovery_steps
        if self.disturbance_initiated:
            info["disturbance_intensity"] = self.disturbance_info.get("intensity", 1.0)
            info["disturbance_type"] = self.disturbance_info.get("type", "unknown")

        if done:
            truncated = reason == "timeout"
            info.update(self._episode_info(reason))
            if self.disturbance_initiated:
                self.recent_recoveries.append(1 if info["tumble_recovered"] else 0)
            self._check_curriculum_advancement()
            return self._get_observation(), float(reward), not truncated, truncated, info
        return self._get_observation(), float(reward), False, False, info

    def _episode_info(self, reason):
        """Everything the training log needs, once per episode."""
        d = self.disturbance_info
        recovered = bool(self.disturbance_recovered and reason != "crash")
        return {"reason": reason, "episode_done": True, "episode_steps": self.episode_steps,
                "episode_return": self.ep["ret"], "curriculum_level": self.curriculum_level,
                "tumble_initiated": bool(self.disturbance_initiated), "tumble_recovered": recovered,
                "recovery_steps": self.recovery_steps if recovered else 0,
                "recovery_time_s": self.recovery_steps / HZ if recovered else 0.0,
                "disturbance_intensity": d.get("intensity", 0.0), "disturbance_type": d.get("type", "none"),
                "impact_turns": d.get("turns", 0.0), "impact_turns_planned": d.get("turns_planned", 0.0),
                "impact_peak_rate": d.get("peak_rate", 0.0), "impact_cut": bool(d.get("cut", False)),
                "min_altitude": self.ep["min_alt"] if self.ep["min_alt"] < 1e8 else self.link.alt,
                "max_angular_velocity": self.ep["max_rate"],
                "final_distance": float(np.linalg.norm(self.link.obs(self.target)[0:2]))}

    def land(self):
        self._active = False
        self._set_wind(np.zeros(3))
        self.link.land(timeout=120)

    def close(self):
        self._stop = True
        self._active = False
        try:
            self._set_wind(np.zeros(3))
            self.disturbance_injector.restore_limits()
            self.link.hold()
        except Exception:
            pass


# Keep same class name for compatibility (as AirSim)
DroneFlipRecoveryEnv = DroneFlipRecoveryEnvGated
