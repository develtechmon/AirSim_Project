"""
AirSim Drone Recovery Environment - FIXED VERSION
==================================================
Properly configured for learning from scratch with curriculum stages.

KEY FIXES:
1. Stage 1 uses altitude-based termination (not collision detection)
2. Reduced reward magnitudes so -100 crash penalty is meaningful
3. Proper max_steps = 2000 for long episodes
4. Small disturbances in Stage 1 to prevent passive hovering
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
    - Stage 1: Learn stable hover (loose bounds, dense rewards, lenient termination)
    - Stage 2: Recover from moderate disturbances (wind, small angular impulses)
    - Stage 3: Recover from severe impacts (flips, tumbles, bird strikes)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 20}
    
    def __init__(
        self,
        stage: int = 1,
        max_steps: int = 2000,
        control_freq: int = 20,
        debug: bool = False
    ):
        super().__init__()
        
        self.stage = stage
        self.max_steps = max_steps
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.debug = debug
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print(f"✅ Connected to AirSim")
        
        # Action space: [roll_rate, pitch_rate, yaw_rate, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )
        
        # Target hover position (NED coordinates)
        self.target_position = np.array([0.0, 0.0, -10.0])
        
        # Stage-specific parameters
        self._configure_stage()
        
        # Episode tracking
        self.current_step = 0
        self._stable_count = 0
        self._episode_count = 0
        self._disturbance_counter = 0
        
        if self.debug:
            print(f"\n🎯 Stage {self.stage} Configuration:")
            print(f"   Max Steps: {self.max_steps}")
            print(f"   Crash Height: {self.crash_height}m")
            print(f"   OOB Radius: {self.out_of_bounds_radius}m")
            print(f"   Disturbance Freq: {self.disturbance_freq}")
    
    def _configure_stage(self):
        """Configure environment parameters based on curriculum stage."""
        if self.stage == 1:
            # Stage 1: Lenient, focus on basic hover learning
            self.crash_height = 0.3  # Very low - almost at ground
            self.out_of_bounds_radius = 50.0  # Very generous
            self.disturbance_freq = 0.01  # Rare small disturbances
            
        elif self.stage == 2:
            # Stage 2: Moderate difficulty
            self.crash_height = 0.5
            self.out_of_bounds_radius = 30.0
            self.disturbance_freq = 0.05
            
        else:  # Stage 3
            # Stage 3: Challenging impacts
            self.crash_height = 0.5
            self.out_of_bounds_radius = 30.0
            self.disturbance_freq = 0.10
    
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
        time.sleep(1.0)
        
        # Stabilize at hover
        self.client.moveByVelocityAsync(0, 0, 0, duration=0.5).join()
        time.sleep(0.5)
        
        # Reset wind to zero
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
        
        # Reset episode counters
        self.current_step = 0
        self._stable_count = 0
        self._episode_count += 1
        self._disturbance_counter = 0
        
        obs = self._get_observation()
        
        if self.debug and self._episode_count <= 3:
            print(f"\n🔄 RESET Episode {self._episode_count}")
            print(f"   Position: [{obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f}]")
            print(f"   Altitude: {-obs[2]:.2f}m")
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
        REDUCED MAGNITUDES to make crash penalty meaningful.
        """
        pos = obs[0:3]
        quat = obs[3:7]  # [qw, qx, qy, qz]
        lin_vel = obs[7:10]
        ang_vel = obs[10:13]
        
        # === Position Error ===
        pos_error = np.linalg.norm(pos - self.target_position)
        pos_reward = -1.0 * pos_error  # Reduced from -2.0
        
        # === Uprightness ===
        uprightness = abs(quat[0])  # |qw|
        upright_reward = 5.0 * uprightness  # Reduced from 15.0
        
        # === Recovery Bonus ===
        if uprightness < 0.7:
            recovery_bonus = 5.0 * uprightness  # Reduced from 20.0
        else:
            recovery_bonus = 3.0  # Reduced from 10.0
        
        # === Altitude Recovery ===
        altitude = -pos[2]
        if altitude < 5.0:
            altitude_bonus = 3.0 * (altitude / 5.0)  # Reduced from 10.0
        else:
            altitude_bonus = 3.0
        
        # === Velocity Penalties ===
        lin_vel_mag = np.linalg.norm(lin_vel)
        ang_vel_mag = np.linalg.norm(ang_vel)
        vel_penalty = -0.3 * lin_vel_mag  # Reduced from -0.5
        ang_penalty = -0.5 * ang_vel_mag  # Reduced from -1.0
        
        # === Stabilization Bonus ===
        if ang_vel_mag > 2.0:
            stab_bonus = 2.0 * (1.0 / (1.0 + ang_vel_mag))  # Reduced from 5.0
        else:
            stab_bonus = 2.0
        
        # === Dense Shaping ===
        stability_bonus = 3.0 * np.exp(-0.5 * pos_error - 0.1 * ang_vel_mag)  # Reduced from 10.0
        
        # === Stage Bonuses ===
        stage_bonus = 0.0
        if self.stage >= 2:
            is_stable = (pos_error < 1.0 and uprightness > 0.95 and ang_vel_mag < 0.5)
            if is_stable:
                stage_bonus = 5.0  # Reduced from 20.0
        
        if self.stage == 3:
            if uprightness < 0.5:
                stage_bonus += 10.0 * uprightness  # Reduced from 30.0
        
        total_reward = (
            pos_reward + 
            upright_reward + 
            recovery_bonus +
            altitude_bonus +
            vel_penalty + 
            ang_penalty + 
            stab_bonus +
            stability_bonus + 
            stage_bonus
        )
        
        if self.debug and self.current_step % 100 == 0:
            print(f"   [Reward] Step {self.current_step}: Total={total_reward:.2f}, Alt={altitude:.2f}m, Pos_err={pos_error:.2f}m")
        
        return total_reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one control step."""
        self.current_step += 1
        
        # Apply disturbances
        if self.current_step % 50 == 0:  # Every ~2.5 seconds
            self._inject_disturbance()
        
        # Scale actions
        roll_rate = float(action[0])  # -1 to 1 rad/s
        pitch_rate = float(action[1])
        yaw_rate = float(action[2])
        throttle = float(np.clip(action[3], 0.0, 1.0))
        
        # Send control command
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=roll_rate,
            pitch_rate=pitch_rate,
            yaw_rate=yaw_rate,
            throttle=throttle,
            duration=self.dt
        )
        time.sleep(self.dt)
        
        # Get new observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(obs)
        
        # === Termination Conditions ===
        terminated = False
        truncated = False
        reason = None
        
        altitude = -obs[2]  # Convert NED to positive-up
        
        # STAGE 1: Use altitude-based termination (more lenient for learning)
        if self.stage == 1:
            if altitude < self.crash_height:
                terminated = True
                reward = -100.0
                reason = "too_low"
                if self.debug:
                    print(f"   ⚠️  TOO LOW at step {self.current_step}: altitude={altitude:.2f}m")
        else:
            # STAGES 2 & 3: Use collision detection
            if self.current_step > 20:  # Grace period
                collision_info = self.client.simGetCollisionInfo()
                if collision_info.has_collided:
                    terminated = True
                    reward = -100.0
                    reason = "ground_collision"
                    if self.debug:
                        print(f"   💥 COLLISION at step {self.current_step}")
        
        # Low altitude penalty (but don't terminate in Stage 1)
        if altitude < 2.0 and not terminated:
            low_alt_penalty = -20.0 * np.exp(-(altitude - 0.3))
            reward += low_alt_penalty
        
        # Out of bounds check
        horizontal_dist = np.linalg.norm(obs[0:2])
        if not terminated and horizontal_dist > self.out_of_bounds_radius:
            terminated = True
            reward = -50.0
            reason = "out_of_bounds"
            if self.debug:
                print(f"   🚫 OUT OF BOUNDS at step {self.current_step}: distance={horizontal_dist:.2f}m")
        
        # Success condition (Stage 1 only)
        if self.stage == 1 and not terminated:
            pos_error = np.linalg.norm(obs[0:3] - self.target_position)
            is_hovering = (
                pos_error < 0.5 and
                abs(obs[3]) > 0.98 and
                np.linalg.norm(obs[7:10]) < 0.3 and
                np.linalg.norm(obs[10:13]) < 0.3
            )
            
            if is_hovering:
                self._stable_count += 1
            else:
                self._stable_count = 0
            
            if self._stable_count >= 50:
                terminated = True
                reward += 100.0
                reason = "hover_success"
                if self.debug:
                    print(f"   ✅ HOVER SUCCESS at step {self.current_step}")
        
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
            "altitude": float(altitude),
            "ang_vel_mag": float(np.linalg.norm(obs[10:13])),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _inject_disturbance(self) -> None:
        """Inject stage-appropriate disturbances."""
        if np.random.rand() > self.disturbance_freq:
            return
        
        state = self.client.simGetGroundTruthKinematics()
        
        if self.stage == 1:
            # Stage 1: Tiny disturbances
            disturbance_type = np.random.choice(['wind', 'sensor_noise'])
            
            if disturbance_type == 'wind':
                state.linear_velocity.x_val += float(np.random.uniform(-0.3, 0.3))
                state.linear_velocity.y_val += float(np.random.uniform(-0.3, 0.3))
                if self.debug:
                    self._disturbance_counter += 1
                    if self._disturbance_counter <= 5:
                        print(f"   [Stage 1] Wind gust applied")
            
            elif disturbance_type == 'sensor_noise':
                state.angular_velocity.x_val += float(np.random.uniform(-0.2, 0.2))
                state.angular_velocity.y_val += float(np.random.uniform(-0.2, 0.2))
                state.angular_velocity.z_val += float(np.random.uniform(-0.1, 0.1))
                if self.debug:
                    self._disturbance_counter += 1
                    if self._disturbance_counter <= 5:
                        print(f"   [Stage 1] Sensor noise applied")
            
            self.client.simSetKinematics(state, ignore_collision=True)
        
        elif self.stage == 2:
            # Stage 2: Moderate disturbances
            disturbance_type = np.random.choice(['wind', 'angular_impulse'], p=[0.6, 0.4])
            
            if disturbance_type == 'wind':
                state.linear_velocity.x_val += float(np.random.uniform(-2.0, 2.0))
                state.linear_velocity.y_val += float(np.random.uniform(-2.0, 2.0))
            
            elif disturbance_type == 'angular_impulse':
                state.angular_velocity.x_val += float(np.random.uniform(-2.0, 2.0))
                state.angular_velocity.y_val += float(np.random.uniform(-2.0, 2.0))
                state.angular_velocity.z_val += float(np.random.uniform(-1.0, 1.0))
            
            self.client.simSetKinematics(state, ignore_collision=True)
        
        else:  # Stage 3
            # Stage 3: Severe impacts
            disturbance_type = np.random.choice(['flip', 'tumble', 'collision'])
            
            if disturbance_type == 'flip':
                state.angular_velocity.y_val += float(np.random.uniform(-6.0, 6.0))
            
            elif disturbance_type == 'tumble':
                state.angular_velocity.x_val += float(np.random.uniform(-4.0, 4.0))
                state.angular_velocity.y_val += float(np.random.uniform(-4.0, 4.0))
                state.angular_velocity.z_val += float(np.random.uniform(-3.0, 3.0))
            
            elif disturbance_type == 'collision':
                state.angular_velocity.x_val += float(np.random.uniform(-3.0, 3.0))
                state.angular_velocity.y_val += float(np.random.uniform(-3.0, 3.0))
                state.linear_velocity.x_val += float(np.random.uniform(-2.0, 2.0))
                state.linear_velocity.y_val += float(np.random.uniform(-2.0, 2.0))
            
            self.client.simSetKinematics(state, ignore_collision=True)
    
    def close(self):
        """Cleanup resources."""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)


if __name__ == "__main__":
    # Quick test
    print("Testing environment...")
    env = AirSimDroneRecoveryEnv(stage=1, debug=True)
    
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial altitude: {-obs[2]:.2f}m")
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            print(f"Step {i}: Reward={reward:.2f}, Alt={-obs[2]:.2f}m")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {i}")
            print(f"Reason: {info['reason']}")
            break
    
    env.close()
    print("✅ Environment test complete")