"""
AirSim Drone Impact Recovery Environment - Production Ready
============================================================
Features:
- Angle rate control (orientation-invariant)
- Realistic impact simulation via simSetKinematics
- Progressive curriculum with gated advancement
- Dense reward shaping for stable learning
- Extensive logging and diagnostics
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from typing import Dict, Tuple, Optional


class AirSimDroneRecoveryEnv(gym.Env):
    """
    PPO-ready environment for drone impact recovery with curriculum learning.
    
    Action Space: [roll_rate, pitch_rate, yaw_rate, throttle] in rad/s
    Observation Space: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13D
    
    Curriculum Stages:
    - Stage 1: Learn stable hover (loose bounds, dense rewards)
    - Stage 2: Recover from moderate disturbances (wind, small angular impulses)
    - Stage 3: Recover from severe impacts (flips, tumbles, bird strikes)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 20}
    
    def __init__(
        self, 
        stage: int = 1,
        max_steps_per_stage: Optional[Dict[int, int]] = None,
        debug: bool = False
    ):
        super().__init__()
        
        # AirSim connection
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Gym spaces
        # Action: [roll_rate, pitch_rate, yaw_rate, throttle]
        self.action_space = spaces.Box(
            low=np.array([-3.0, -3.0, -2.0, 0.0]),
            high=np.array([3.0, 3.0, 2.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        
        # Stage configuration
        self.stage = stage
        self.max_steps = (max_steps_per_stage or {
            1: 2000,  # Stage 1: Long episodes for hover learning
            2: 1500,  # Stage 2: Medium episodes for disturbance handling
            3: 1000,  # Stage 3: Shorter episodes for impact recovery
        })[stage]
        
        self.current_step = 0
        self.debug = debug
        
        # Target hover position (NED frame)
        self.target_position = np.array([0.0, 0.0, -10.0], dtype=np.float32)
        
        # Disturbance probabilities per stage
        self.disturbance_prob = {1: 0.0, 2: 0.05, 3: 0.10}[stage]
        
        # Stage-specific termination thresholds
        self.crash_height = {1: 0.5, 2: 1.0, 3: 1.5}[stage]  # More lenient in early stages
        self.out_of_bounds_radius = {1: 30.0, 2: 25.0, 3: 20.0}[stage]
        
        # Episode statistics
        self._episode_count = 0
        self._stable_count = 0
        self._last_10_returns = []
        
        print(f"🚁 AirSim Drone Recovery Environment Initialized")
        print(f"   Stage: {stage}")
        print(f"   Max Steps: {self.max_steps}")
        print(f"   Disturbance Probability: {self.disturbance_prob}")
        print(f"   Crash Height Threshold: {self.crash_height}m")
        print(f"   OOB Radius: {self.out_of_bounds_radius}m")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial hover state."""
        super().reset(seed=seed)
        
        # Reset AirSim
        self.client.reset()
        time.sleep(0.5)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Takeoff to hover position
        self.client.takeoffAsync().join()
        time.sleep(1.0)
        
        # Move to target hover position
        self.client.moveToPositionAsync(0, 0, -10, velocity=2).join()
        time.sleep(0.5)
        
        # Reset wind to zero
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
        
        # Reset episode counters
        self.current_step = 0
        self._stable_count = 0
        self._episode_count += 1
        
        obs = self._get_observation()
        
        if self.debug and self._episode_count <= 3:
            print(f"\n🔄 RESET Episode {self._episode_count}")
            print(f"   Position: [{obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f}]")
            print(f"   Orientation (qw): {obs[3]:.3f}")
        
        return obs, {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        state = self.client.getMultirotorState()
        
        return np.array([
            # Position (NED, meters)
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val,
            # Orientation (quaternion: w, x, y, z)
            state.kinematics_estimated.orientation.w_val,
            state.kinematics_estimated.orientation.x_val,
            state.kinematics_estimated.orientation.y_val,
            state.kinematics_estimated.orientation.z_val,
            # Linear velocity (NED, m/s)
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val,
            # Angular velocity (body frame, rad/s)
            state.kinematics_estimated.angular_velocity.x_val,
            state.kinematics_estimated.angular_velocity.y_val,
            state.kinematics_estimated.angular_velocity.z_val,
        ], dtype=np.float32)
    
    def _compute_reward(self, obs: np.ndarray) -> float:
        """
        Dense reward function encouraging hover stability and recovery.
        
        Reward components:
        1. Position error penalty (primary objective)
        2. Uprightness bonus (critical for recovery)
        3. Velocity penalties (stability)
        4. Angular velocity penalties (smooth control)
        5. Dense shaping bonus (guides exploration)
        6. Stage-specific recovery bonus
        """
        pos = obs[0:3]
        quat = obs[3:7]  # [qw, qx, qy, qz]
        lin_vel = obs[7:10]
        ang_vel = obs[10:13]
        
        # === Position Error ===
        pos_error = np.linalg.norm(pos - self.target_position)
        pos_reward = -2.0 * pos_error  # Strong penalty for being far from target
        
        # === Uprightness (Critical for recovery from flips) ===
        # qw close to ±1 means drone is upright
        # qw close to 0 means drone is sideways/inverted
        uprightness = abs(quat[0])  # |qw|
        upright_reward = 15.0 * uprightness  # High weight for staying upright
        
        # === Velocity Penalties (Encourage stability) ===
        lin_vel_mag = np.linalg.norm(lin_vel)
        ang_vel_mag = np.linalg.norm(ang_vel)
        vel_penalty = -0.5 * lin_vel_mag
        ang_penalty = -1.0 * ang_vel_mag
        
        # === Dense Shaping Bonus (Exponential reward for getting close) ===
        # This helps with exploration in early stages
        stability_bonus = 10.0 * np.exp(-0.5 * pos_error - 0.1 * ang_vel_mag)
        
        # === Stage-Specific Bonuses ===
        stage_bonus = 0.0
        if self.stage >= 2:
            # Bonus for being stable after disturbance
            is_stable = (pos_error < 1.0 and uprightness > 0.95 and ang_vel_mag < 0.5)
            if is_stable:
                stage_bonus = 20.0
        
        if self.stage == 3:
            # Extra bonus for recovering from severe orientations
            if uprightness < 0.7:  # Was in bad orientation
                # Reward improvement in orientation
                stage_bonus += 10.0 * uprightness
        
        total_reward = (
            pos_reward + 
            upright_reward + 
            vel_penalty + 
            ang_penalty + 
            stability_bonus + 
            stage_bonus
        )
        
        if self.debug and self.current_step % 100 == 0:
            print(f"   [Step {self.current_step}] Reward Breakdown:")
            print(f"      Pos: {pos_reward:.2f} | Upright: {upright_reward:.2f} | Vel: {vel_penalty:.2f}")
            print(f"      Ang: {ang_penalty:.2f} | Stability: {stability_bonus:.2f} | Stage: {stage_bonus:.2f}")
            print(f"      Total: {total_reward:.2f}")
        
        return total_reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one control step.
        
        Args:
            action: [roll_rate, pitch_rate, yaw_rate, throttle] in rad/s and [0,1]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Extract action components
        roll_rate, pitch_rate, yaw_rate, throttle = action
        
        # Apply control command
        self.client.moveByAngleRatesThrottleAsync(
            float(roll_rate),
            float(pitch_rate),
            float(yaw_rate),
            float(throttle),
            duration=0.05  # 20Hz control rate
        ).join()
        
        # Inject disturbances (if stage >= 2)
        if self.current_step > 100:  # Give initial stabilization time
            if np.random.rand() < self.disturbance_prob:
                self._inject_disturbance()
        
        # Get new observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(obs)
        
        # Update step counter
        self.current_step += 1
        
        # === Termination Conditions ===
        terminated = False
        truncated = False
        reason = None
        
        # Check for crash (too close to ground)
        if obs[2] > -self.crash_height:  # z > -threshold (remember NED: down is positive)
            terminated = True
            reward = -100.0
            reason = "crash"
        
        # Check out of bounds (horizontal drift)
        horizontal_dist = np.linalg.norm(obs[0:2])
        if horizontal_dist > self.out_of_bounds_radius:
            terminated = True
            reward = -50.0
            reason = "out_of_bounds"
        
        # Check for successful hover (Stage 1 only)
        if self.stage == 1:
            pos_error = np.linalg.norm(obs[0:3] - self.target_position)
            is_hovering = (
                pos_error < 0.5 and
                abs(obs[3]) > 0.98 and  # qw > 0.98 (very upright)
                np.linalg.norm(obs[7:10]) < 0.3 and  # low linear velocity
                np.linalg.norm(obs[10:13]) < 0.3  # low angular velocity
            )
            
            if is_hovering:
                self._stable_count += 1
            else:
                self._stable_count = 0
            
            # Success if hovering for 50 consecutive steps (~2.5 seconds)
            if self._stable_count >= 50:
                terminated = True
                reward += 100.0  # Success bonus
                reason = "hover_success"
        
        # Timeout
        if self.current_step >= self.max_steps:
            truncated = True
            if reason is None:
                reason = "timeout"
        
        info = {
            "reason": reason,
            "step": self.current_step,
            "pos_error": float(np.linalg.norm(obs[0:3] - self.target_position)),
            "uprightness": float(abs(obs[3])),
            "ang_vel_mag": float(np.linalg.norm(obs[10:13])),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _inject_disturbance(self) -> None:
        """Inject stage-appropriate disturbances."""
        if self.stage == 1:
            # Stage 1: No disturbances (pure hover learning)
            return
        
        state = self.client.simGetGroundTruthKinematics()
        
        if self.stage == 2:
            # Stage 2: Moderate disturbances
            disturbance_type = np.random.choice(['wind', 'angular_impulse'], p=[0.6, 0.4])
            
            if disturbance_type == 'wind':
                # Wind gust
                wind = airsim.Vector3r(
                    float(np.random.uniform(-5, 5)),
                    float(np.random.uniform(-5, 5)),
                    float(np.random.uniform(-2, 2))
                )
                self.client.simSetWind(wind)
                if self.debug:
                    print(f"   💨 Wind gust: [{wind.x_val:.1f}, {wind.y_val:.1f}, {wind.z_val:.1f}] m/s")
            
            else:  # angular_impulse
                # Small angular velocity injection
                state.angular_velocity.x_val += float(np.random.uniform(-2, 2))
                state.angular_velocity.y_val += float(np.random.uniform(-2, 2))
                state.angular_velocity.z_val += float(np.random.uniform(-1, 1))
                self.client.simSetKinematics(state, ignore_collision=True)
                if self.debug:
                    print(f"   🌀 Angular impulse applied")
        
        elif self.stage == 3:
            # Stage 3: Severe impacts (flips, collisions, bird strikes)
            impact_type = np.random.choice(['flip', 'spin', 'tumble', 'collision'])
            
            if impact_type == 'flip':
                # Rapid pitch rotation (bird strike from front)
                state.angular_velocity.y_val = float(np.random.uniform(-8, 8))
                impact_str = f"Flip (pitch: {state.angular_velocity.y_val:.1f} rad/s)"
            
            elif impact_type == 'spin':
                # Rapid yaw rotation (propeller strike)
                state.angular_velocity.z_val = float(np.random.uniform(-6, 6))
                impact_str = f"Spin (yaw: {state.angular_velocity.z_val:.1f} rad/s)"
            
            elif impact_type == 'tumble':
                # Multi-axis tumble (collision)
                state.angular_velocity.x_val = float(np.random.uniform(-5, 5))
                state.angular_velocity.y_val = float(np.random.uniform(-5, 5))
                state.angular_velocity.z_val = float(np.random.uniform(-3, 3))
                impact_str = "Tumble (multi-axis)"
            
            else:  # collision
                # Combined rotational + translational disturbance
                state.angular_velocity.x_val = float(np.random.uniform(-4, 4))
                state.angular_velocity.y_val = float(np.random.uniform(-4, 4))
                state.linear_velocity.x_val += float(np.random.uniform(-3, 3))
                state.linear_velocity.y_val += float(np.random.uniform(-3, 3))
                impact_str = "Collision (rotation + translation)"
            
            self.client.simSetKinematics(state, ignore_collision=True)
            
            if self.debug:
                print(f"   💥 Impact: {impact_str}")
    
    def set_stage(self, new_stage: int) -> None:
        """Update curriculum stage."""
        if new_stage not in [1, 2, 3]:
            raise ValueError(f"Invalid stage: {new_stage}. Must be 1, 2, or 3.")
        
        self.stage = new_stage
        self.max_steps = {1: 2000, 2: 1500, 3: 1000}[new_stage]
        self.disturbance_prob = {1: 0.0, 2: 0.05, 3: 0.10}[new_stage]
        self.crash_height = {1: 0.5, 2: 1.0, 3: 1.5}[new_stage]
        self.out_of_bounds_radius = {1: 30.0, 2: 25.0, 3: 20.0}[new_stage]
        
        print(f"\n🔄 Stage Updated → {new_stage}")
        print(f"   Max Steps: {self.max_steps}")
        print(f"   Disturbance Prob: {self.disturbance_prob}")
    
    def close(self) -> None:
        """Cleanup resources."""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)