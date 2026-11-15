"""
Disturbance Injector for Impact Recovery Training (ArduPilot SITL + AirSim)
===========================================================================
Same job and same interface as your AirSim disturbance_injector.py
(DisturbanceType, DisturbanceInjector.inject_disturbance(type, intensity) -> info dict),
but the impact is made through ArduPilot itself, so it works the same in SITL and on the real drone:

    BODY-RATE commands (SET_ATTITUDE_TARGET, rates only) - "tumble this fast" - kept until
    the gyro has counted the planned number of turns (verified with impact_check.py).

    FLIP         roll OR pitch at 16 rad/s (SITL quad tops out ~14.4)   turns = 2.0 x intensity
    SPIN         yaw at 10 rad/s                                         turns = 2.0 x intensity
    BIRD_ATTACK  side push (velocity kick 0.15 s, like your force) then
                 roll + pitch + yaw 10/10/6 rad/s                        turns = 1.0 x intensity

    intensity = your curriculum multiplier: EASY 0.7-0.9, MEDIUM 0.9-1.1, HARD 1.1-1.5
    -> flip/spin 1.4-1.8 / 1.8-2.2 / 2.2-3.0 turns, bird 0.7-0.9 / 0.9-1.1 / 1.1-1.5 turns

Impact mode: ArduPilot's rotation-start limits (ATC_ACCEL_R/P/Y_MAX, or ATC_ACC_* on newer
firmware) are set to 0 (= no limit) ONLY during the impact and put back right after,
so the recovery always runs with stock ArduPilot limits.

Safety inside the impact: the tumble is cut early if the drone could no longer stop above
the virtual ground (height - braking distance < floor + margin).
"""

import math
import time
from enum import Enum

import numpy as np

from ap_link import RateKeeper


class DisturbanceType(Enum):
    """Types of disturbances (same names as the AirSim version)"""
    BIRD_ATTACK = "bird_attack"
    FLIP = "flip"
    SPIN = "spin"


LIMIT_NAMES = [["ATC_ACCEL_R_MAX", "ATC_ACCEL_P_MAX", "ATC_ACCEL_Y_MAX"],     # ArduPilot 4.4-4.6
               ["ATC_ACC_R_MAX", "ATC_ACC_P_MAX", "ATC_ACC_Y_MAX"]]           # newer firmware
HZ = 50.0
CAP = 3.0                  # impact never longer than CAP x planned time
BRAKE = 5.0                # m/s^2 braking assumed for the cut (conservative)


class DisturbanceInjector:
    """Injects disturbances through ArduPilot rate commands."""

    def __init__(self, link, rng=None, floor=5.0, margin=2.0, impact_mode=True):
        self.link = link
        self.rng = rng if rng is not None else np.random.RandomState()
        self.floor, self.margin = floor, margin
        self.params = {
            "flip": {"rate": 16.0, "turns": 2.0, "axis": ["roll", "pitch"]},
            "spin": {"rate": 10.0, "turns": 2.0},
            "bird_attack": {"rates": (10.0, 10.0, 6.0), "turns": 1.0,
                            "push_speed": (3.0, 8.0), "push_s": 0.15},   # your force 30-80 N x 0.1
        }
        self.limits = self._read_limits() if impact_mode else {}

    # ------------------------------------------------------------ helpers
    def _read_limits(self):
        for names in LIMIT_NAMES:
            got = self.link.get_params(names)
            if len(got) == len(names):
                return got
        raise RuntimeError("Could not read ATC_ACCEL / ATC_ACC limits from the FC")

    def restore_limits(self):
        if self.limits:
            self.link.send_params(self.limits)

    def _sign(self):
        return 1.0 if self.rng.rand() < 0.5 else -1.0

    def _brake_floor(self):
        v_down = max(0.0, float(self.link.vel[2]))
        return self.link.alt - v_down ** 2 / (2 * BRAKE)

    def _tumble(self, p, q, r, turns_goal):
        """Rate commands until the gyro counted turns_goal turns. Returns (turns, seconds, cut, peak_rate)."""
        planned = turns_goal * 2 * math.pi / float(np.linalg.norm([p, q, r]))
        if self.limits:
            self.link.send_params({k: 0 for k in self.limits})          # impact mode on
        rk = RateKeeper(HZ)
        turned, peak, cut = 0.0, 0.0, False
        t0 = t_prev = time.time()
        try:
            while turned < turns_goal and time.time() - t0 < CAP * planned:
                self.link.send_rates(p, q, r, 0.5)
                rk.wait()
                self.link.update()
                now = time.time()
                rate = float(np.linalg.norm(self.link.rates))
                turned += rate * (now - t_prev) / (2 * math.pi)
                peak = max(peak, rate)
                t_prev = now
                if self._brake_floor() < self.floor + self.margin:
                    cut = True
                    break
        finally:
            self.restore_limits()                                         # stock ArduPilot again
        return turned, time.time() - t0, cut, peak

    # ------------------------------------------------------------ API (same as AirSim version)
    def inject_disturbance(self, disturbance_type, intensity=1.0):
        if disturbance_type == DisturbanceType.FLIP:
            return self._apply_flip(intensity)
        if disturbance_type == DisturbanceType.SPIN:
            return self._apply_spin(intensity)
        if disturbance_type == DisturbanceType.BIRD_ATTACK:
            return self._apply_bird_attack(intensity)
        raise ValueError(f"Unknown disturbance type: {disturbance_type}")

    def _apply_flip(self, intensity):
        prm = self.params["flip"]
        axis = self.rng.choice(prm["axis"])
        rate = prm["rate"] * self._sign()
        p, q = (rate, 0.0) if axis == "roll" else (0.0, rate)
        goal = prm["turns"] * intensity
        turns, secs, cut, peak = self._tumble(p, q, 0.0, goal)
        return {"type": "flip", "axis": str(axis), "angular_velocity": abs(rate) * 180 / math.pi,
                "intensity": float(intensity), "turns_planned": goal, "turns": turns,
                "impact_s": secs, "peak_rate": peak, "cut": cut}

    def _apply_spin(self, intensity):
        prm = self.params["spin"]
        direction = self._sign()
        goal = prm["turns"] * intensity
        turns, secs, cut, peak = self._tumble(0.0, 0.0, prm["rate"] * direction, goal)
        return {"type": "spin", "angular_velocity": prm["rate"] * direction * 180 / math.pi,
                "direction": "clockwise" if direction > 0 else "counterclockwise",
                "intensity": float(intensity), "turns_planned": goal, "turns": turns,
                "impact_s": secs, "peak_rate": peak, "cut": cut}

    def _apply_bird_attack(self, intensity):
        prm = self.params["bird_attack"]
        direction = self.rng.choice(["left", "right", "front", "back"])
        speed = self.rng.uniform(*prm["push_speed"]) * intensity
        vx, vy = {"front": (speed, 0.0), "back": (-speed, 0.0),
                  "left": (0.0, -speed), "right": (0.0, speed)}[str(direction)]
        rk = RateKeeper(HZ)
        t_end = time.time() + prm["push_s"]
        while time.time() < t_end:                                         # the hit pushes sideways
            self.link.send_velocity(vx, vy, 0.0)
            rk.wait()
        self.link.update()
        p, q, r = (v * self._sign() for v in prm["rates"])
        goal = prm["turns"] * intensity
        turns, secs, cut, peak = self._tumble(p, q, r, goal)
        return {"type": "bird_attack", "direction": str(direction), "push_speed": speed,
                "angular_velocity": float(np.linalg.norm([p, q, r])) * 180 / math.pi,
                "intensity": float(intensity), "turns_planned": goal, "turns": turns,
                "impact_s": secs + prm["push_s"], "peak_rate": peak, "cut": cut}

    def get_random_disturbance(self):
        return self.rng.choice(list(DisturbanceType))
