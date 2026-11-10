"""
DRONE HOVER DISTURBANCE ENVIRONMENT (ArduPilot SITL + AirSim, 13 observations)
==============================================================================
Same job as your AirSim drone_hover_disturbance_env.py:
hold position at the target while random wind gusts push the drone.

Same wind as the original:
    random direction, strength 0..20 m/s (vertical part x0.3), new gust every 1-3 s
    AirSim wind only adds air drag, and the default drone has little drag, so the AirSim
    setting is NOT real-world m/s. Measured with wind_check.py (drone holding position):
        5 m/s -> 0.7 deg lean, 3 cm drift  |  20 m/s -> 14 deg lean, 0.6-0.7 m drift at each gust
    So training uses gusts 0..20 (AirSim setting). Match the real fan by LEAN ANGLE, not m/s.
    applied with AirSim simSetWind (AirSim still does the physics with ArduPilot)

Same network interface as Stage 1 (so the Stage 1 weights fit exactly):
    observation (13): position minus target, velocity, quaternion, body rates
                      scaled with Stage 1's obs_norm.npz, clipped to +/-10
    action (3):       velocity command N, E, D (m/s), sent in GUIDED mode

Episode:
    reset  = wind off, the teacher flies the drone to a random start
             within +/-1 m of home (+/-0.3 m height), then wind on
    target = home, 2.0 m high (fixed)
    500 steps at 20 Hz = 25 s, like the original
    ends early (failure) if: > 4 m from target, height < 0.5 m or > 5 m, tilt > 60 deg

Reward (small numbers on purpose - the original's huge rewards gave value loss ~1e6):
    +1 alive  - distance (m)  - 0.1 x speed (m/s)  - 0.01 x |action|^2 ;  -50 on failure

Safety pilot: while PPO updates its network (several seconds, no commands),
a background thread turns the wind off and holds the drone at the target.
The episode that was interrupted is cut ("update_pause") - not counted as success or failure.

Stock ArduPilot baseline: env.step(None) sends the TARGET POSITION instead of a velocity, so
ArduPilot's own position controller (the same one PosHold/Loiter use) holds the point - no network.
Used by test_stage2_policy.py / collect_stage2_statistics.py with --model ardupilot.

Self-test (AirSim + SITL in GPS mode running):
    py .\\drone_hover_disturbance_env.py
    -> 2 episodes flown by the Stage 1 network, prints wind and distance every 50 steps
"""

import threading
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pymavlink import mavutil

from ap_link import APLink, RateKeeper
from pid_expert import PIDExpertHover

HZ = 20.0
ALT = 2.0                  # m, hover height (same as Stage 1)
START_XY = 1.0             # m, random start offset from home
START_Z = 0.3              # m
VMAX_XY_SEND = 2.0         # m/s, safety limit on what is sent to ArduPilot
VMAX_Z_SEND = 1.0          # m/s
FAIL_DIST = 4.0            # m from target
FAIL_ALT_MIN = 0.5
FAIL_ALT_MAX = 5.0
FAIL_TILT = 60.0           # deg
FAIL_PENALTY = 50.0
CLIP_OBS = 10.0
ACT_LIMIT = 5.0            # same action space as the Stage 1 transfer check
RESET_FLY_S = 8.0          # max time to fly to the start point
STABLE_M = 0.5             # "holding" = within this distance of the target
RUNS = Path("runs")
_T = mavutil.mavlink
POS_MASK = (_T.POSITION_TARGET_TYPEMASK_VX_IGNORE | _T.POSITION_TARGET_TYPEMASK_VY_IGNORE |
            _T.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
            _T.POSITION_TARGET_TYPEMASK_AX_IGNORE | _T.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            _T.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            _T.POSITION_TARGET_TYPEMASK_YAW_IGNORE)    # use position x,y,z + yaw rate (0)


def run_folder(run=None, new=False):
    """Stage 2 run folders: runs/run1, runs/run2, ... (highest number = latest)."""
    nums = [int(p.name[3:]) for p in RUNS.glob("run*") if p.name[3:].isdigit()]
    if run is not None:
        return RUNS / (run if run == "practice" else f"run{int(run)}")
    if new or not nums:
        return RUNS / f"run{max(nums, default=0) + 1}"
    return RUNS / f"run{max(nums)}"


def stage1_folder(path=None):
    """Stage 1 run with the trained network (default: newest ../Stage_1/runs/run<N>)."""
    if path:
        return Path(path)
    root = Path("..") / "Stage_1" / "runs"
    if not root.exists() and (Path("..") / "stage_1" / "runs").exists():
        root = Path("..") / "stage_1" / "runs"
    nums = [int(p.name[3:]) for p in root.glob("run*") if p.name[3:].isdigit()]
    if not nums:
        raise SystemExit(f"No Stage 1 run found in {root} - give --stage1-run <path>")
    return root / f"run{max(nums)}"


class WindSchedule:
    """Same gust model as the original: new random wind every 1-3 s."""

    def __init__(self, max_wind, rng, fixed=None):
        self.max_wind, self.rng, self.fixed = max_wind, rng, fixed
        self.timer, self.interval = 0.0, rng.uniform(1.0, 3.0)
        self.current = self._new()

    def _new(self):
        theta = self.rng.uniform(0, 2 * np.pi)
        if self.fixed is not None:                       # constant strength, sideways
            return np.array([self.fixed * np.cos(theta), self.fixed * np.sin(theta), 0.0])
        phi = self.rng.uniform(0, np.pi)
        s = self.rng.uniform(0, self.max_wind)
        return np.array([s * np.sin(phi) * np.cos(theta), s * np.sin(phi) * np.sin(theta),
                         s * np.cos(phi) * 0.3])

    def step(self, dt):
        """Returns True when the wind changed."""
        if self.fixed is not None:
            return False
        self.timer += dt
        if self.timer >= self.interval:
            self.current = self._new()
            self.timer, self.interval = 0.0, self.rng.uniform(1.0, 3.0)
            return True
        return False


class DroneHoverDisturbanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_norm_path, max_steps=500, wind_strength=20.0, fixed_wind=None,
                 seed=None, connect="tcp:127.0.0.1:5762", debug=False):
        super().__init__()
        norm = np.load(obs_norm_path)
        self.mean, self.std = norm["mean"], norm["std"]
        self.max_steps, self.wind_strength, self.fixed_wind = max_steps, wind_strength, fixed_wind
        self.debug = debug
        self.observation_space = spaces.Box(-CLIP_OBS, CLIP_OBS, (13,), np.float32)
        self.action_space = spaces.Box(-ACT_LIMIT, ACT_LIMIT, (3,), np.float32)
        self.rng = np.random.RandomState(seed)

        import airsim
        self._airsim = airsim
        self.sim = airsim.MultirotorClient()
        self.sim.confirmConnection()
        self.link = APLink(connect)
        self.link.wait_state()
        self.teacher = PIDExpertHover()
        self.home = None
        self.target = None
        self.rk = RateKeeper(HZ)
        self.wind = None
        self.steps = 0
        self.ep_count = 0
        self._ep = {}

        # safety pilot thread
        self._last_step_t = time.time()
        self._active = False
        self._took_over = False
        self._stop = False
        self._thread = threading.Thread(target=self._safety_pilot, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ helpers
    def _set_wind(self, w):
        self.sim.simSetWind(self._airsim.Vector3r(float(w[0]), float(w[1]), float(w[2])))

    def _send_position(self, p):
        """Stock ArduPilot baseline: GUIDED position target -> ArduPilot's own position controller
        (the same one PosHold/Loiter use) holds the point. Heading held (yaw rate 0), like send_velocity."""
        with self.link.io_lock:
            self.link.m.mav.set_position_target_local_ned_send(
                0, self.link.m.target_system, self.link.m.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED, POS_MASK,
                float(p[0]), float(p[1]), float(p[2]), 0, 0, 0, 0, 0, 0, 0, 0.0)

    def _obs_raw(self):
        return self.link.obs(self.target)

    def _norm(self, o):
        return np.clip((o - self.mean) / self.std, -CLIP_OBS, CLIP_OBS).astype(np.float32)

    def _teacher_cmd(self, goal):
        o = self.link.obs(goal)
        return self.teacher.get_action({"position": o[0:3], "velocity": o[3:6]})

    def _safety_pilot(self):
        """Holds the drone (wind off) whenever no step came for > 0.15 s (PPO update)."""
        sim = None
        while not self._stop:
            time.sleep(0.05)
            if not self._active or time.time() - self._last_step_t < 0.15:
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
            self.link.send_velocity(*self._teacher_cmd(self.target))

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "seed" in options:                      # identical start + wind for a fair comparison
            self.rng = np.random.RandomState(options["seed"])
        if "fixed_wind" in options:                # None = gusts 0..max, number = constant strength
            self.fixed_wind = options["fixed_wind"]
        self._active = False                       # safety pilot off while we fly here
        self._took_over = False
        self._set_wind(np.zeros(3))
        self.link.update()
        if not self.link.armed:
            self.link.takeoff_nogps(ALT)
        if self.home is None:
            self.link.update()
            self.home = self.link.pos.copy()
            print(f"[env] home: N {self.home[0]:.2f} E {self.home[1]:.2f} m, target height {ALT} m")
        self.target = np.array([self.home[0], self.home[1], -ALT])
        start = self.target + np.array([self.rng.uniform(-START_XY, START_XY),
                                        self.rng.uniform(-START_XY, START_XY),
                                        self.rng.uniform(-START_Z, START_Z)])
        rk = RateKeeper(HZ)
        t_end = time.time() + RESET_FLY_S
        while time.time() < t_end:
            self.link.update()
            o = self.link.obs(start)
            if np.linalg.norm(o[0:3]) < 0.15 and np.linalg.norm(o[3:6]) < 0.2:
                break
            self.link.send_velocity(*self.teacher.get_action({"position": o[0:3], "velocity": o[3:6]}))
            rk.wait()
        self.wind = WindSchedule(self.wind_strength, self.rng, self.fixed_wind)
        self._set_wind(self.wind.current)
        self.steps = 0
        self.ep_count += 1
        self._ep = {"dist": [], "hdist": [], "wind": [], "ret": 0.0}
        self.rk = RateKeeper(HZ)
        self.link.update()
        obs = self._obs_raw()
        self._last_step_t = time.time()
        self._active = True
        return self._norm(obs), {"start_dist": float(np.linalg.norm(obs[0:3]))}

    def step(self, action):
        if self._took_over:                          # PPO paused: cut this episode
            self._took_over = False
            self._last_step_t = time.time()
            self.link.update()
            info = self._finish("update_pause")
            return self._norm(self._obs_raw()), 0.0, False, True, info

        if action is None:                           # stock ArduPilot: hold the target by itself
            a = np.zeros(3)
            self._send_position(self.target)
        else:
            a = np.clip(np.asarray(action, dtype=np.float64), -ACT_LIMIT, ACT_LIMIT)
            send = np.array([np.clip(a[0], -VMAX_XY_SEND, VMAX_XY_SEND),
                             np.clip(a[1], -VMAX_XY_SEND, VMAX_XY_SEND),
                             np.clip(a[2], -VMAX_Z_SEND, VMAX_Z_SEND)])
            self.link.send_velocity(*send)
        if self.wind.step(1.0 / HZ):
            self._set_wind(self.wind.current)
        self.rk.wait()
        self._last_step_t = time.time()
        self.link.update()
        self.steps += 1

        obs = self._obs_raw()
        dist = float(np.linalg.norm(obs[0:3]))          # 3D distance to target (reward, safety)
        hdist = float(np.linalg.norm(obs[0:2]))         # horizontal distance (= "Dist" in the paper)
        speed = float(np.linalg.norm(obs[3:6]))
        wind_mag = float(np.linalg.norm(self.wind.current))
        reward = 1.0 - dist - 0.1 * speed - 0.01 * float(np.sum(a ** 2))
        self._ep["dist"].append(dist)
        self._ep["hdist"].append(hdist)
        self._ep["wind"].append(wind_mag)

        reason = None
        if dist > FAIL_DIST:
            reason = "out_of_bounds"
        elif self.link.alt < FAIL_ALT_MIN or self.link.alt > FAIL_ALT_MAX:
            reason = "altitude_violation"
        elif self.link.tilt_deg > FAIL_TILT:
            reason = "tilt"
        terminated = reason is not None
        if terminated:
            reward -= FAIL_PENALTY
        truncated = (not terminated) and self.steps >= self.max_steps
        if truncated:
            reason = "timeout"
        self._ep["ret"] += reward

        info = {"distance": hdist, "distance_3d": dist, "altitude": float(self.link.alt),
                "wind_magnitude": wind_mag, "reason": reason or "running"}
        if terminated or truncated:
            info.update(self._finish(reason))
            if self.debug:
                print(f"[env] episode end: {reason} | steps {self.steps} | "
                      f"mean dist {info['ep_mean_dist']:.2f} m | mean wind {info['ep_mean_wind']:.1f} m/s")
        elif self.debug and self.steps % 50 == 0:
            print(f"[env] step {self.steps:3d} | dist {dist:.2f} m | wind {wind_mag:.1f} m/s")
        return self._norm(obs), float(reward), terminated, truncated, info

    def _finish(self, reason):
        d = np.array(self._ep["dist"]) if self._ep.get("dist") else np.array([np.nan])
        h = np.array(self._ep["hdist"]) if self._ep.get("hdist") else np.array([np.nan])
        w = np.array(self._ep["wind"]) if self._ep.get("wind") else np.array([np.nan])
        return {"reason": reason, "ep_steps": self.steps, "ep_return": self._ep.get("ret", 0.0),
                "ep_dist_trace": [round(x, 3) for x in self._ep.get("dist", [])],
                "ep_mean_dist": float(np.nanmean(d)), "ep_max_dist": float(np.nanmax(d)),
                "ep_mean_hdist": float(np.nanmean(h)), "ep_max_hdist": float(np.nanmax(h)),
                "ep_end_dist": float(np.nanmean(d[-20:])),
                "ep_rmse": float(np.sqrt(np.nanmean(d ** 2))),
                "ep_hold_pct": float(100.0 * np.nanmean(d < STABLE_M)),
                "ep_mean_wind": float(np.nanmean(w)), "ep_max_wind": float(np.nanmax(w)),
                "episode_done": True}

    def close(self):
        self._stop = True
        self._active = False
        try:
            self._set_wind(np.zeros(3))
            self.link.hold()
        except Exception:
            pass

    def land(self):
        self._active = False
        self._set_wind(np.zeros(3))
        self.link.land()


if __name__ == "__main__":
    import argparse

    from stable_baselines3 import PPO

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-run", default=None, help="Stage 1 run folder (default: newest)")
    ap.add_argument("--episodes", type=int, default=2)
    args = ap.parse_args()
    s1 = stage1_folder(args.stage1_run)
    print(f"Self-test with the Stage 1 network from {s1}")
    env = DroneHoverDisturbanceEnv(s1 / "models" / "obs_norm.npz", seed=0, debug=True)
    model = PPO.load(str(s1 / "models" / "stage1_ppo_init.zip"), device="cpu")
    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            print(f"\nEpisode {ep + 1}: start {info['start_dist']:.2f} m from target, "
                  f"first wind {np.linalg.norm(env.wind.current):.1f} m/s")
            while True:
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(act)
                if term or trunc:
                    break
            print(f"Result: {info['reason']} | mean dist {info['ep_mean_dist']:.2f} m | "
                  f"max {info['ep_max_dist']:.2f} m | held within {STABLE_M} m "
                  f"{info['ep_hold_pct']:.0f}% of the time | mean wind {info['ep_mean_wind']:.1f} m/s")
        print("\nIf the distance grew while the wind was on, AirSim wind works with ArduPilot.")
    finally:
        env.land()
        env.close()
