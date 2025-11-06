# airsim_env.py
import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

class AirSimDroneEnv(gym.Env):
    """
    PPO-ready AirSim quad environment for hover learning (Stage 1).
    Features:
        - PWM control in [0, 1] range
        - Continuous shaping rewards for altitude stability
        - Loosened success condition for faster learning
        - Small hover bias to counter gravity
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, stage=1, hover_bias=0.05, debug=False):
        super().__init__()

        # ─── AirSim connection ─────────────────────────────────────────────
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # ─── Gym spaces ────────────────────────────────────────────────────
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # ─── Environment parameters ────────────────────────────────────────
        self.stage = stage
        self.hover_bias = float(hover_bias)
        self.max_steps = 500
        self.current_step = 0
        self.debug = debug

        # ─── Disturbance settings (used later in curriculum) ───────────────
        self.disturbance_probability = {1: 0.02, 2: 0.10, 3: 0.15}[stage]
        self.rotor_format = self._detect_rotor_format()

        print(f"🎮 Env initialized — Stage {self.stage} | Hover bias={self.hover_bias} | Disturb p={self.disturbance_probability}")

    # ───────────────────────────────────────────────────────────────────────
    def _detect_rotor_format(self):
        test = self.client.getRotorStates()
        if hasattr(test, "rotors") and len(test.rotors) > 0:
            return "dict" if isinstance(test.rotors[0], dict) else "object"
        return "unknown"

    # ───────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Takeoff and hover
        self.client.takeoffAsync().join()
        time.sleep(1.5)
        self.client.moveToPositionAsync(0, 0, -10, 1).join()
        time.sleep(1.5)

        self.current_step = 0
        self._stable_count = 0

        obs = self._get_observation()
        if self.debug or not hasattr(self, "_reset_count"):
            self._reset_count = 1
        else:
            self._reset_count += 1

        if self._reset_count <= 3:
            print(f"\n  🔄 RESET #{self._reset_count}: "
                  f"pos=({obs[0]:.2f},{obs[1]:.2f},{obs[2]:.2f}) qw={obs[3]:.3f}")

        return obs, {}

    # ───────────────────────────────────────────────────────────────────────
    def _get_observation(self):
        s = self.client.getMultirotorState()
        return np.array([
            s.kinematics_estimated.position.x_val,
            s.kinematics_estimated.position.y_val,
            s.kinematics_estimated.position.z_val,
            s.kinematics_estimated.orientation.w_val,
            s.kinematics_estimated.orientation.x_val,
            s.kinematics_estimated.orientation.y_val,
            s.kinematics_estimated.orientation.z_val,
            s.kinematics_estimated.linear_velocity.x_val,
            s.kinematics_estimated.linear_velocity.y_val,
            s.kinematics_estimated.linear_velocity.z_val,
            s.kinematics_estimated.angular_velocity.x_val,
            s.kinematics_estimated.angular_velocity.y_val,
            s.kinematics_estimated.angular_velocity.z_val,
        ], dtype=np.float32)

    # ───────────────────────────────────────────────────────────────────────
    def _compute_reward(self, obs):
        """Reward function tuned for hover stability."""

        pos = obs[0:3]
        qw = obs[3]
        lin_vel = obs[7:10]
        ang_vel = obs[10:13]

        target = np.array([0, 0, -10], dtype=np.float32)
        pos_error = np.linalg.norm(pos - target)
        vel_mag = np.linalg.norm(lin_vel)
        ang_mag = np.linalg.norm(ang_vel)

        # Reward shaping terms
        pos_reward = -pos_error                      # penalize distance
        orient_reward = 10.0 * qw                    # upright orientation
        vel_pen = -0.1 * vel_mag                     # slow motion
        ang_pen = -0.5 * ang_mag                     # stable rotation
        alt_pen = -0.2 * abs(obs[2] + 10.0)          # penalize altitude error
        stability_bonus = 5.0 * np.exp(-pos_error)   # dense shaping reward

        total = pos_reward + orient_reward + vel_pen + ang_pen + alt_pen + stability_bonus

        if self.debug and self.current_step < 3:
            print(f"    [Reward Breakdown] pos={pos_reward:.2f} orient={orient_reward:.2f} "
                  f"vel={vel_pen:.2f} ang={ang_pen:.2f} alt={alt_pen:.2f} bonus={stability_bonus:.2f} → total={total:.2f}")

        return total

    # ───────────────────────────────────────────────────────────────────────
    def step(self, action):
        """Execute one control step using PWM commands."""
        debug = (self.current_step < 5 and getattr(self, "_reset_count", 1) <= 2)

        # Map PPO output [-1, 1] → PWM [0, 1]
        motor_pwms = (np.asarray(action, dtype=np.float32) + 1.0) * 0.5
        motor_pwms += self.hover_bias
        motor_pwms = np.clip(motor_pwms, 0.0, 1.0)

        if debug:
            print(f"    [STEP {self.current_step}] Action={action} → PWM={motor_pwms}")

        # Apply command
        self.client.moveByMotorPWMsAsync(
            float(motor_pwms[0]),
            float(motor_pwms[1]),
            float(motor_pwms[2]),
            float(motor_pwms[3]),
            duration=0.1
        ).join()

        obs = self._get_observation()
        reward = self._compute_reward(obs)

        # Termination conditions
        self.current_step += 1
        terminated, truncated = False, False
        reason = None

        # Define stable hover check
        def is_stable(o):
            z_ok = abs(o[2] + 10.0) < 0.5
            up_ok = o[3] > 0.97
            v_ok = np.linalg.norm(o[7:10]) < 0.7
            ang_ok = np.linalg.norm(o[10:13]) < 0.7
            return z_ok and up_ok and v_ok and ang_ok

        if is_stable(obs):
            self._stable_count += 1
        else:
            self._stable_count = 0

        # Success if stable for 10 consecutive steps (~1s)
        if self.stage == 1 and self._stable_count >= 10:
            terminated = True
            reason = "success"
            reward += 20.0

        # Crash if too low
        if not terminated and obs[2] > -0.5:
            terminated = True
            reason = "crash"
            reward = -100.0

        # Out of bounds horizontally
        if not terminated and np.linalg.norm(obs[0:2]) > 20.0:
            terminated = True
            reason = "oob"
            reward = -50.0

        # Timeout
        if self.current_step >= self.max_steps:
            truncated = True
            if reason is None:
                reason = "time"

        info = {"terminal_reason": reason, "terminated": terminated, "truncated": truncated}

        if debug:
            print(f"    [STEP {self.current_step}] Reward={reward:.2f} Term={terminated} Reason={reason}\n")

        return obs, reward, terminated, truncated, info

    # ───────────────────────────────────────────────────────────────────────
    def _inject_stage_appropriate_disturbance(self):
        """Used in later curriculum stages."""
        pose = self.client.simGetVehiclePose()
        if self.stage == 1:
            pose.position.x_val += np.random.uniform(-1, 1)
            pose.position.y_val += np.random.uniform(-1, 1)
        elif self.stage == 2:
            pose.position.x_val += np.random.uniform(-3, 3)
            pose.position.y_val += np.random.uniform(-3, 3)
        elif self.stage == 3:
            pose.position.x_val += np.random.uniform(-5, 5)
            pose.position.y_val += np.random.uniform(-5, 5)
        self.client.simSetVehiclePose(pose, True)

    def set_stage(self, stage):
        self.stage = stage
        self.disturbance_probability = {1: 0.02, 2: 0.10, 3: 0.15}[stage]
        print(f"🔄 Stage → {stage}")
