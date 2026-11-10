r"""
PPO RECOVERY ENVIRONMENT - PERFORMANCE-GATED CURRICULUM  (Stage 3 step 3c, ArduPilot SITL + AirSim)
==================================================================================================

1. HOW TO RUN
------------------------------------------------------------------------
    This file is not run by itself - the training script uses it:

    py .\train_stage_3_gated_curriculum.py

    Quick self-test of this environment (5 episodes with the BC network, no learning, ~2 min,
    AirSim + SITL in GPS mode running):

    py .\drone_flip_recovery_env_injector_gated.py --episodes 5


2. EXPECTED RESULT  (self-test)
------------------------------------------------------------------------
    One line per episode, e.g.
    Episode 3 | flip EASY 0.82 | turns 1.6 | lowest 22.6 m | RECOVERED in 0.5 s | return 9.1
    With the BC network: all 5 RECOVERED, lowest heights like bc_impact_check (flip 16-23 m,
    bird 21-27 m, spin ~30 m), returns roughly -15 ... +15.


3. ONE EPISODE = ONE HIT + ONE RECOVERY
------------------------------------------------------------------------
    reset()   Step 1  ArduPilot flies back to the start point (30 m) and holds 1 s   (ArduPilot)
              Step 2  the HIT: type + intensity from the curriculum level            (impact_check hit)
              Step 3  rotation limits (ATC_ACCEL) stay OFF, GUID_OPTIONS bit 3 ON    ("thrust as thrust")
    step(a)   every 20 ms: PPO's action -> ATTITUDE target + thrust -> ArduPilot 400 Hz loop
    end       CALM      rate < 1.2 rad/s, tilt < 30 deg, falling < 1 m/s   -> RECOVERED, back to ArduPilot
              CRASH     below the 5 m virtual ground / cannot stop above it -> FAILED, script rescues
              TIMEOUT   not calm after 8 s                                  -> FAILED, attitude-rule finishes
              PAUSE     PPO stopped sending (network update)               -> attitude-rule finishes,
                                                                              episode NOT counted
    After every episode ATC_ACCEL and GUID_OPTIONS are put back to YOUR values.


4. CURRICULUM  (same as AirSim Stage 3 and impact_check)
------------------------------------------------------------------------
    Level        Intensity    Turns flip / spin / bird      Advance when (last 50, min 50 episodes)
    0 EASY       0.7 - 0.9    1.4-1.8 / 1.4-1.8 / 0.7-0.9   80 % recovered
    1 MEDIUM     0.9 - 1.1    1.8-2.2 / 1.8-2.2 / 0.9-1.1   70 % recovered
    2 HARD       1.1 - 1.5    2.2-3.0 / 2.2-3.0 / 1.1-1.5   60 % target
    Hit types: flip 45 % | bird 45 % | spin 10 %   (same mix as collect_demonstration.py / BC)


5. OBSERVATION AND ACTION  (identical to collect_demonstration.py and the BC network)
------------------------------------------------------------------------
    OBS (13)  down_x down_y down_z | p q r | v_down | dalt_since_hit | v_fwd v_right | previous action
              scaled with models\obs_norm.npz (the BC scaling, fixed)
    ACT (3)   roll rad | pitch rad (heading frame) | thrust x hover
              clipped to roll/pitch +-0.6 rad and thrust 0 ... 1/hover (= 0 ... 100 %)


6. REWARD  (per 20 ms step; numbers are small on purpose, like Stage 2)
------------------------------------------------------------------------
    Part                       Value                          Pushes PPO to ...
    height lost this step      -2.0 per m going DOWN          lose less height (thesis claim)
    time                       -0.05 per step (-2.5 per s)    recover faster
    spin                       -0.01 x rate (rad/s)           arrest the rotation (phase 1)
    not upright                -0.02 x (1 - cos tilt)         get upright (phase 2)
    sideways speed             -0.01 x speed (m/s)            less drift
    jerky commands             -0.02 x |action change|^2      smooth commands (real motors)
    CALM reached               +20                            finish the recovery
    CRASH (virtual ground)     -50                            never crash
    TIMEOUT (8 s)              -10
    Only going DOWN is penalised (no reward for climbing), so PPO cannot earn points by delaying CALM.


7. DOMAIN RANDOMISATION  (training only - tests switch it off)
------------------------------------------------------------------------
    sensor noise    gyro 0.05 rad/s, velocity 0.1 m/s, height 0.1 m, attitude 0.01   (x --obs-noise)
    thrust error    commanded thrust x random 0.95 ... 1.05 per episode              (--thrust-rand)
                    (= a wrong MOT_THST_HOVER on the real drone)
    wind gusts      AirSim wind 0 ... --wind-strength m/s, new gust every 1-3 s       (needs airsim)


8. SAFETY
------------------------------------------------------------------------
    - parameters: refuses to start if the FC still holds test values (ATC_ACCEL 0 / GUID_OPTIONS 8)
    - rescue: below the virtual ground the script saves the drone (full-thrust rate-rule)
    - safety pilot: during PPO updates a background thread flies attitude-rule, so the drone is never
      left without commands
    - REAL crash (disarmed / on the floor) -> RuntimeError: training saves a checkpoint and stops;
      restart SITL, then  py .\train_stage_3_gated_curriculum.py --resume
"""

import argparse
import math
import threading
import time
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import impact_check as ic
from ap_link import APLink, RateKeeper

HZ = ic.HZ                       # 50 Hz, same as collect_demonstration / BC
FLOOR = ic.FLOOR                 # 5 m virtual ground
BRAKE = ic.BRAKE                 # 8 m/s^2 (impact cut)
START_ALT = ic.ALT               # 30 m
CLIP_OBS = 10.0
ACT_LIMIT = 5.0                  # SB3 action box (same as train_imitation); real limits in ic.action_to_attitude
MAX_RECOVERY_S = 8.0             # TIMEOUT
PAUSE_S = 0.15                   # no step for this long = PPO update -> safety pilot takes over
FINISH_MAX_S = 10.0              # attitude-rule / rescue finishing time limit
LEVEL_NAMES = ["EASY", "MEDIUM", "HARD"]
LEVEL_KEYS = ["easy", "medium", "hard"]
LEVEL_RANGES = [ic.CURRICULUM[k] for k in LEVEL_KEYS]
THRESHOLDS = {0: 0.80, 1: 0.70, 2: 0.60}
TYPES = {"flip": 0.45, "bird": 0.45, "spin": 0.10}
TYPE_NAMES = {"flip": "flip", "bird": "bird_attack", "spin": "spin"}     # names in the AirSim log
NOISE = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0, 0, 0], dtype=np.float32)
# reward weights (section 6)
W_DOWN, W_TIME, W_SPIN, W_TILT, W_SIDE, W_SMOOTH = 2.0, 0.05, 0.01, 0.02, 0.01, 0.02
R_CALM, R_CRASH, R_TIMEOUT = 20.0, -50.0, -10.0


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

    def __init__(self, obs_norm_path=ic.OBS_NORM, wind_strength=0.0, obs_noise=0.0, thrust_rand=0.0,
                 seed=None, debug=False, connect="tcp:127.0.0.1:5762", curriculum_frozen=False, start_level=0,
                 max_recovery_s=MAX_RECOVERY_S):
        super().__init__()
        norm = np.load(obs_norm_path)
        self.mean, self.std = norm["mean"], norm["std"]
        self.wind_strength, self.obs_noise, self.thrust_rand = wind_strength, obs_noise, thrust_rand
        self.debug, self.max_steps = debug, int(max_recovery_s * HZ)
        self.action_space = spaces.Box(-ACT_LIMIT, ACT_LIMIT, (3,), np.float32)
        self.observation_space = spaces.Box(-CLIP_OBS, CLIP_OBS, (13,), np.float32)
        self.rng = np.random.RandomState(seed)

        self.sim = None
        if wind_strength > 0:
            try:
                import airsim
                self._airsim = airsim
                self.sim = airsim.MultirotorClient()
                self.sim.confirmConnection()
            except Exception as e:
                print(f"   (AirSim wind not available: {e} - training WITHOUT wind)")
                self.sim, self.wind_strength = None, 0.0
        self.link = APLink(connect, rate_hz=50)
        self.link.wait_state()
        self.limits = ic.read_limits(self.link)
        got = self.link.get_params(["GUID_OPTIONS", "MOT_THST_HOVER"])
        if not self.limits or "GUID_OPTIONS" not in got:
            raise SystemExit("Could not read ATC_ACCEL / GUID_OPTIONS from the FC - stopping")
        self.guid, self.hover = int(got["GUID_OPTIONS"]), float(got.get("MOT_THST_HOVER", 0.4))
        if any(abs(v) < 1e-6 for v in self.limits.values()) or self.guid & ic.GUID_THRUST_BIT:
            raise SystemExit(f"STOP - the FC still has test values from a crashed run: {self.limits} "
                             f"GUID_OPTIONS {self.guid}\nFix in MAVProxy: param set ATC_ACCEL_R_MAX 110000 / "
                             "ATC_ACCEL_P_MAX 110000 / ATC_ACCEL_Y_MAX 27000 / GUID_OPTIONS 0")
        self.target = None
        self.episode_count = 0
        self.wind = WindSchedule(0.0, self.rng)
        self._reset_tracking()

        # performance-gated curriculum (same rule as AirSim)
        self.curriculum_level = start_level
        self.curriculum_frozen = curriculum_frozen
        self.level_names = LEVEL_NAMES
        self.recent_recoveries = deque(maxlen=50)
        self.advancement_thresholds = THRESHOLDS
        self.min_episodes_per_level = 50
        self.episodes_at_current_level = 0
        self.level_advanced = False
        self.advancement_info = {}
        self.forced = None                                   # tests: force (type, intensity) per episode

        # safety pilot (PPO update pauses)
        self._lock = threading.Lock()
        self._last_step_t = time.time()
        self._active = False
        self._took_over = False
        self._stop = False
        self._thread = threading.Thread(target=self._safety_pilot, daemon=True)
        self._thread.start()
        if self.debug:
            print("✓ PPO recovery environment (ArduPilot, attitude + thrust)")
            print(f"  - Start level: {LEVEL_NAMES[self.curriculum_level]} | start {START_ALT:.0f} m | "
                  f"virtual ground {FLOOR:.0f} m | hover thrust {self.hover:.2f}")
            print(f"  - Randomisation: obs noise x{self.obs_noise} | thrust +-{self.thrust_rand * 100:.0f} % | "
                  f"wind 0-{self.wind_strength} m/s")

    # ------------------------------------------------------------ helpers
    def _reset_tracking(self):
        self.ep = {"min_alt": 1e9, "max_rate": 0.0, "ret": 0.0, "type": "none", "level_key": "easy",
                   "intensity": 0.0, "turns": 0.0, "turns_planned": 0.0, "peak_rate": 0.0, "cut": False,
                   "alt0": START_ALT, "alt_end_impact": START_ALT, "rescued": False}
        self.steps = 0
        self.last = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.prev_alt = START_ALT
        self.thrust_k = 1.0
        self.recovered = False
        self.recovery_steps = 0

    def _set_wind(self, w):
        if self.sim is None:
            return
        try:
            self.sim.simSetWind(self._airsim.Vector3r(float(w[0]), float(w[1]), float(w[2])))
        except Exception:
            pass

    def _obs(self):
        raw = ic.bc_observation(self.link, self.ep["alt0"], self.last)
        if self.obs_noise > 0:
            raw = raw + self.rng.normal(0.0, 1.0, 13).astype(np.float32) * NOISE * self.obs_noise
        return np.clip((raw - self.mean) / self.std, -CLIP_OBS, CLIP_OBS).astype(np.float32)

    def _check_real_crash(self):
        if ic.crashed(self.link):
            self._restore()
            raise RuntimeError("💥 REAL CRASH (disarmed or on the floor) - restart SITL, then --resume")

    def _restore(self):
        ic.send_params(self.link, {**dict(self.limits), "GUID_OPTIONS": self.guid})

    def _doomed(self):
        v_down = max(0.0, float(self.link.vel[2]))
        return self.link.alt < FLOOR or self.link.alt - v_down ** 2 / (2 * ic.RESCUE_BRAKE) < ic.RESCUE_FLOOR

    def _calm(self):
        return (float(np.linalg.norm(self.link.rates)) < ic.RATE_OK and self.link.tilt_deg < ic.TILT_OK
                and float(self.link.vel[2]) < 1.0)

    def _finish(self, rescue):
        """attitude-rule (or full-thrust rescue) until calm, then YOUR parameters back. Blocking."""
        rk = RateKeeper(HZ)
        t0 = time.time()
        while time.time() - t0 < FINISH_MAX_S:
            self.link.update()
            self._check_real_crash()
            if self._calm() and not self._doomed():
                break
            rescue = rescue or self._doomed()
            if rescue:
                self.link.send_rates(*ic.rate_rule(self.link, self.hover, full=True))
            else:
                ic.send_attitude(self.link, *ic.attitude_rule(self.link, self.hover))
            rk.wait()
            self.ep["min_alt"] = min(self.ep["min_alt"], self.link.alt)
        self._restore()
        self.link.send_velocity(0.0, 0.0, 0.0)
        return rescue

    def _safety_pilot(self):
        """PPO update = no step for > PAUSE_S: fly attitude-rule until calm, then hold still."""
        while not self._stop:
            time.sleep(0.02)
            if not self._active or time.time() - self._last_step_t < PAUSE_S:
                continue
            if not self._lock.acquire(blocking=False):
                continue
            try:
                if self._active and time.time() - self._last_step_t >= PAUSE_S:
                    if not self._took_over:
                        self._took_over = True
                        self._set_wind(np.zeros(3))
                        self.ep["rescued"] = self._finish(rescue=False) or self.ep["rescued"]
                    else:
                        self.link.update()
                        self.link.send_velocity(0.0, 0.0, 0.0)
            except Exception as e:                           # never kill the thread silently
                print(f"   (safety pilot: {e})")
            finally:
                self._lock.release()

    def _pick_hit(self):
        if self.forced is not None:
            kind, intensity = self.forced
        else:
            kinds = list(TYPES)
            p = np.array(list(TYPES.values()))
            kind = kinds[self.rng.choice(len(kinds), p=p / p.sum())]
            intensity = self.rng.uniform(*LEVEL_RANGES[self.curriculum_level])
        return kind, float(intensity)

    def _hit(self, kind, intensity):
        """The impact - identical to impact_check / collect_demonstration (rates until the gyro counted the turns)."""
        p, q, r = ic.rates_for(kind, self.rng)
        goal = ic.BASE_TURNS[kind] * intensity
        planned = goal * 2 * math.pi / float(np.linalg.norm([p, q, r]))
        ic.send_params(self.link, {k: 0 for k in self.limits})          # impact mode
        rk = RateKeeper(HZ)
        t0 = t_prev = time.time()
        turned, peak, cut = 0.0, 0.0, False
        while turned < goal and time.time() - t0 < ic.CAP * planned:
            self.link.send_rates(p, q, r, 0.5)
            rk.wait()
            self.link.update()
            now = time.time()
            rate = float(np.linalg.norm(self.link.rates))
            turned += rate * (now - t_prev) / (2 * math.pi)
            t_prev, peak = now, max(peak, rate)
            if ic.brake_floor(self.link) < FLOOR + ic.MARGIN:
                cut = True
                break
            self._check_real_crash()
        ic.send_params(self.link, {"GUID_OPTIONS": self.guid | ic.GUID_THRUST_BIT})   # rotation limits stay OFF
        self.ep.update(turns=turned, turns_planned=goal, peak_rate=peak, cut=cut, alt_end_impact=self.link.alt)

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
                                     "recovery_rate": rate, "threshold": threshold, "episode": self.episode_count}
            print("\n" + "=" * 70)
            print("🎓 CURRICULUM ADVANCEMENT!")
            print("=" * 70)
            print(f"   Level {old} ({LEVEL_NAMES[old]}) MASTERED!")
            print(f"   Recovery rate: {rate * 100:.1f}% (needed {threshold * 100:.0f}%)")
            print(f"   Advancing to Level {self.curriculum_level} ({LEVEL_NAMES[self.curriculum_level]})")
            print("   💾 MODEL WILL BE AUTO-SAVED!")
            print("=" * 70 + "\n")

    def set_curriculum_state(self, level, episodes_at_level, recent):
        """--resume: rebuild the curriculum from the episode log."""
        self.curriculum_level = int(level)
        self.episodes_at_current_level = int(episodes_at_level)
        self.recent_recoveries.clear()
        self.recent_recoveries.extend(int(x) for x in recent[-50:])

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "seed" in options:
            self.rng.seed(options["seed"])
        with self._lock:
            self._active = False
            self._took_over = False
            self._set_wind(np.zeros(3))
            self.link.update()
            if not self.link.armed:
                self.link.takeoff_nogps(START_ALT)
            self._check_real_crash()
            if self.target is None:
                self.link.update()
                self.target = self.link.pos.copy()
                self.target[2] = -START_ALT
            self._restore()                                    # always start from YOUR parameters
            ic.fly_back(self.link, self.target, 1.0)
            self._reset_tracking()
            kind, intensity = self._pick_hit()
            self.ep.update(type=kind, intensity=intensity, level_key=LEVEL_KEYS[self.curriculum_level],
                           alt0=self.link.alt)
            if self.thrust_rand > 0:
                self.thrust_k = float(self.rng.uniform(1 - self.thrust_rand, 1 + self.thrust_rand))
            self.episode_count += 1
            self.episodes_at_current_level += 1
            if self.debug:
                print(f"   ⚠️  Hit {self.episode_count}: {kind} {LEVEL_NAMES[self.curriculum_level]} "
                      f"x{intensity:.2f} | episodes at level {self.episodes_at_current_level}")
            self._hit(kind, intensity)
            self.wind = WindSchedule(self.wind_strength, self.rng)
            self._set_wind(self.wind.current)
            self.link.update()
            self.prev_alt = self.link.alt
            self.ep["min_alt"] = self.link.alt
            self.rk = RateKeeper(HZ)
            obs = self._obs()
            self._last_step_t = time.time()
            self._active = True
        return obs, {}

    def step(self, action):
        with self._lock:
            return self._step(action)

    def _step(self, action):
        if self._took_over:                                   # PPO paused: episode cut, not counted
            self._took_over = False
            self._active = False
            self._last_step_t = time.time()
            self.episodes_at_current_level = max(0, self.episodes_at_current_level - 1)   # not counted
            return self._obs(), 0.0, False, True, self._episode_info("update_pause")

        self.steps += 1
        q, thr, a = ic.action_to_attitude(np.asarray(action, dtype=np.float64), self.link, self.hover)
        ic.send_attitude(self.link, q, float(np.clip(thr * self.thrust_k, 0.0, 1.0)))
        if self.wind.step(1.0 / HZ):
            self._set_wind(self.wind.current)
        self.rk.wait()
        self._last_step_t = time.time()
        self.link.update()
        self._check_real_crash()

        alt = self.link.alt
        rate = float(np.linalg.norm(self.link.rates))
        up_z = math.cos(math.radians(self.link.tilt_deg))
        hspeed = float(np.linalg.norm(self.link.vel[0:2]))
        self.ep["min_alt"] = min(self.ep["min_alt"], alt)
        self.ep["max_rate"] = max(self.ep["max_rate"], rate)
        reward = (-W_DOWN * max(0.0, self.prev_alt - alt) - W_TIME - W_SPIN * min(rate, 20.0)
                  - W_TILT * (1.0 - up_z) - W_SIDE * min(hspeed, 10.0)
                  - W_SMOOTH * float(np.sum((a - self.last) ** 2)))
        self.prev_alt = alt
        self.last = a

        reason = None
        if self._doomed():
            reward += R_CRASH
            reason = "crash"
            self.ep["rescued"] = self._finish(rescue=True)
        elif self._calm():
            reward += R_CALM
            reason = "recovered"
            self.recovered = True
            self.recovery_steps = self.steps
            self._restore()                                   # back to ArduPilot (velocity hold)
            self.link.send_velocity(0.0, 0.0, 0.0)
        elif self.steps >= self.max_steps:
            reward += R_TIMEOUT
            reason = "timeout"
            self.ep["rescued"] = self._finish(rescue=False)
        self.ep["ret"] += reward
        if reason is None:
            return self._obs(), float(reward), False, False, {"curriculum_level": self.curriculum_level}

        self._active = False
        self._set_wind(np.zeros(3))
        info = self._episode_info(reason)
        self.recent_recoveries.append(1 if info["tumble_recovered"] else 0)
        self._check_curriculum_advancement()
        if self.debug:
            print(f"   {'✅ RECOVERED' if self.recovered else '❌ ' + reason.upper()} | "
                  f"{self.steps / HZ:.2f} s | lowest {self.ep['min_alt']:.1f} m | return {self.ep['ret']:.1f}")
        return self._obs(), float(reward), reason != "timeout", reason == "timeout", info

    def _episode_info(self, reason):
        """Everything the training log needs, once per episode (same keys as the AirSim log + extras)."""
        e = self.ep
        rec = bool(self.recovered and reason == "recovered")
        self.link.update()
        return {"reason": reason, "episode_done": True, "episode_steps": self.steps,
                "episode_return": e["ret"], "curriculum_level": self.curriculum_level,
                "tumble_initiated": True, "tumble_recovered": rec,
                "recovery_steps": self.recovery_steps if rec else 0,
                "recovery_time_s": self.recovery_steps / HZ if rec else 0.0,
                "disturbance_intensity": e["intensity"], "disturbance_type": TYPE_NAMES.get(e["type"], e["type"]),
                "impact_turns": e["turns"], "impact_turns_planned": e["turns_planned"],
                "impact_peak_rate": e["peak_rate"], "impact_cut": bool(e["cut"]),
                "min_altitude": e["min_alt"] if e["min_alt"] < 1e8 else self.link.alt,
                "height_loss_m": e["alt0"] - (e["min_alt"] if e["min_alt"] < 1e8 else self.link.alt),
                "alt_end_impact_m": e["alt_end_impact"], "max_angular_velocity": e["max_rate"],
                "final_distance": float(np.linalg.norm(self.link.pos[0:2] - self.target[0:2])),
                "rescued": bool(e["rescued"]), "thrust_k": self.thrust_k}

    def land(self):
        self._active = False
        self._set_wind(np.zeros(3))
        self._restore()
        self.link.land(timeout=120)

    def close(self):
        self._stop = True
        self._active = False
        try:
            self._set_wind(np.zeros(3))
            self._restore()
            self.link.send_velocity(0.0, 0.0, 0.0)
        except Exception:
            pass


# Same class name as AirSim (the training / test scripts import this)
DroneFlipRecoveryEnv = DroneFlipRecoveryEnvGated


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Self-test: fly a few episodes with the BC network (no learning)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--level", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--conn", default="tcp:127.0.0.1:5762")
    a = ap.parse_args()
    import torch
    net = ic.load_policy_net("models/recovery_policy_best.pth")
    env = DroneFlipRecoveryEnv(curriculum_frozen=True, start_level=a.level, seed=0, connect=a.conn)
    try:
        for n in range(1, a.episodes + 1):
            obs, _ = env.reset()
            done = trunc = False
            while not (done or trunc):
                with torch.no_grad():
                    act = net(torch.as_tensor(obs[None]))[0].numpy()
                obs, r, done, trunc, info = env.step(act)
            print(f"Episode {n} | {info['disturbance_type']} {LEVEL_NAMES[a.level]} {info['disturbance_intensity']:.2f} | "
                  f"turns {info['impact_turns']:.1f} | lowest {info['min_altitude']:.1f} m | "
                  + (f"RECOVERED in {info['recovery_time_s']:.1f} s" if info["tumble_recovered"] else info["reason"].upper())
                  + f" | return {info['episode_return']:.1f}")
    finally:
        env.land()
        env.close()
