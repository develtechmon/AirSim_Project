"""
POST-IMPACT RECOVERY ENVIRONMENT - FIXED FOR CONTINUOUS CONTROL
Key insight: Your keyboard controller works because it sends commands continuously
without waiting. This version does the same.
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from enum import Enum
from typing import Dict, Tuple, Optional
from collections import deque


class TrainingStage(Enum):
    HOVER = "hover"
    DISTURBANCE = "disturbance"
    IMPACT = "impact"


class PostImpactRecoveryEnv(gym.Env):
    """
    Continuous control - matches working keyboard controller pattern
    """
    
    metadata = {'render.modes': []}
    
    def __init__(
        self,
        training_stage: str = "hover",
        max_episode_steps: int = 500,
        spawn_altitude: float = 15.0,
        enable_logging: bool = False,
    ):
        super().__init__()
        
        self.training_stage = TrainingStage(training_stage)
        self.time_step = 0.05  # 50ms like your working code
        self.max_episode_steps = max_episode_steps
        self.spawn_altitude = spawn_altitude
        self.enable_logging = enable_logging
        
        # Recovery thresholds
        self.stable_tilt_threshold = 0.08
        self.stable_angular_vel_threshold = 0.15
        self.stable_duration_required = 40
        
        # Ground limits
        self.ground_altitude_limit = 0.5
        self.crash_velocity_threshold = -2.0
        self.max_altitude = 30.0
        
        # Action: 4 motor PWM [0,1]
        # Transform will map to [0.40, 0.80] centered at 0.60
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Observation: 35 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32
        )
        
        # AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # State
        self.current_step = 0
        self.episode_reward = 0
        self.previous_action = np.array([0.5, 0.5, 0.5, 0.5])  # Will transform to 0.60
        
        # History
        self.tilt_history = deque(maxlen=5)
        self.angular_vel_history = deque(maxlen=5)
        
        # Tracking
        self.impact_occurred = False
        self.steps_since_impact = 0
        self.peak_tilt = 0.0
        self.peak_angular_vel = 0.0
        
        self.stable_steps = 0
        self.recovery_time = None
        self.lowest_altitude = 999.0
        
        self.total_episodes = 0
        self.successful_recoveries = 0
        self.crashes = 0
        
        # Stage configuration
        self._configure_stage()
    
    def _configure_stage(self):
        """Configure environment based on training stage"""
        if self.training_stage == TrainingStage.HOVER:
            self.apply_disturbance = False
            self.max_episode_steps = 300
            
        elif self.training_stage == TrainingStage.DISTURBANCE:
            self.apply_disturbance = True
            self.max_episode_steps = 400
            
        elif self.training_stage == TrainingStage.IMPACT:
            self.apply_disturbance = True
            self.max_episode_steps = 500
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        
        self.client.reset()
        time.sleep(0.5)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(0.5)
        
        # Conservative PWM values (from user's proven diagnostic code)
        HOVER_PWM = 0.63
        TAKEOFF_PWM = 0.68
        
        # Takeoff using user's proven PWM
        for _ in range(40):  # 2 seconds at 50ms
            self.client.moveByMotorPWMsAsync(TAKEOFF_PWM, TAKEOFF_PWM, TAKEOFF_PWM, TAKEOFF_PWM, self.time_step)
            time.sleep(self.time_step)
        
        # Move to altitude
        self.client.moveToPositionAsync(0, 0, -self.spawn_altitude, 3).join()
        time.sleep(1.0)
        
        # Reset state
        self.current_step = 0
        self.episode_reward = 0
        self.previous_action = np.array([0.5, 0.5, 0.5, 0.5])
        
        for _ in range(5):
            self.tilt_history.append(0.0)
            self.angular_vel_history.append(0.0)
        
        self.impact_occurred = False
        self.steps_since_impact = 0
        self.peak_tilt = 0.0
        self.peak_angular_vel = 0.0
        
        self.stable_steps = 0
        self.recovery_time = None
        self.lowest_altitude = self.spawn_altitude
        
        self.total_episodes += 1
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray):
        action = np.clip(action, 0.0, 1.0)
        
        # Transform [0,1] to center around 0.63 (user's proven hover PWM)
        # Map to [0.43, 0.83] with center at 0.63
        action_transformed = 0.43 + (action * 0.40)
        
        # Send command WITHOUT .join() - continuous control
        self.client.moveByMotorPWMsAsync(
            float(action_transformed[0]),  # FR
            float(action_transformed[1]),  # RL
            float(action_transformed[2]),  # FL
            float(action_transformed[3]),  # RR
            self.time_step
        )
        
        # Short sleep (like clock.tick)
        time.sleep(self.time_step * 0.8)
        
        self.previous_action = action
        
        # Apply disturbance if needed
        if self.apply_disturbance and not self.impact_occurred:
            if self.current_step == 30:
                if self.training_stage == TrainingStage.DISTURBANCE:
                    self._apply_small_disturbance()
                elif self.training_stage == TrainingStage.IMPACT:
                    self._apply_impact()
        
        if self.impact_occurred:
            self.steps_since_impact += 1
        
        # Get observation and compute reward
        obs = self._get_observation()
        reward = self._compute_reward(obs)
        terminated, truncated, info = self._check_termination(obs)
        
        self.current_step += 1
        self.episode_reward += reward
        
        info.update(self._get_info())
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current state"""
        state = self.client.getMultirotorState()
        k = state.kinematics_estimated
        
        # Velocities
        vel = k.linear_velocity
        linear_vel = np.array([vel.x_val, vel.y_val, vel.z_val])
        
        ang_vel = k.angular_velocity
        angular_vel = np.array([ang_vel.x_val, ang_vel.y_val, ang_vel.z_val])
        angular_vel_mag = np.linalg.norm(angular_vel)
        
        # Orientation
        ori = k.orientation
        orientation_quat = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])
        
        # Acceleration
        acc = k.linear_acceleration
        linear_acc = np.array([acc.x_val, acc.y_val, acc.z_val])
        
        # Euler angles
        roll, pitch, yaw = airsim.to_eularian_angles(ori)
        euler_angles = np.array([roll, pitch, yaw])
        tilt_angle = np.sqrt(roll**2 + pitch**2)
        
        # Track peaks
        self.peak_tilt = max(self.peak_tilt, tilt_angle)
        self.peak_angular_vel = max(self.peak_angular_vel, angular_vel_mag)
        
        # Update history
        self.tilt_history.append(tilt_angle)
        self.angular_vel_history.append(angular_vel_mag)
        
        # Position
        pos = k.position
        altitude = -pos.z_val
        vertical_vel = -vel.z_val
        
        self.lowest_altitude = min(self.lowest_altitude, altitude)
        
        # Impact indicators
        impact_flag = 1.0 if self.impact_occurred else 0.0
        time_since_impact = min(self.steps_since_impact / 100.0, 1.0)
        
        # Build observation
        obs = np.concatenate([
            linear_vel,              # 3
            angular_vel,             # 3
            orientation_quat,        # 4
            linear_acc,              # 3
            euler_angles,            # 3
            [altitude],              # 1
            [vertical_vel],          # 1
            self.previous_action,    # 4
            [impact_flag],           # 1
            [time_since_impact],     # 1
            [tilt_angle],            # 1
            [angular_vel_mag],       # 1
            list(self.tilt_history),        # 5
            list(self.angular_vel_history)  # 5
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, obs):
        """Clean reward function"""
        reward = 0.0
        
        roll, pitch = obs[12:14]
        tilt = np.sqrt(roll**2 + pitch**2)
        angular_vel_mag = obs[24]
        altitude = obs[15]
        vertical_vel = obs[16]
        impact_flag = obs[21]
        
        # Core: Minimize tilt and spin (squared penalties)
        reward -= 10.0 * (tilt ** 2)
        reward -= 5.0 * (angular_vel_mag ** 2)
        
        # Stability bonus
        is_stable = (tilt < self.stable_tilt_threshold and 
                    angular_vel_mag < self.stable_angular_vel_threshold)
        
        if is_stable:
            reward += 20.0
            self.stable_steps += 1
            
            if self.stable_steps > 20:
                reward += 10.0
        else:
            self.stable_steps = 0
        
        # Recovery success
        if (impact_flag > 0.5 and 
            self.stable_steps >= self.stable_duration_required and
            self.recovery_time is None):
            
            reward += 200.0
            self.recovery_time = self.steps_since_impact
            
            if self.recovery_time < 50:
                reward += 50.0
            elif self.recovery_time < 100:
                reward += 25.0
        
        # Altitude penalties
        if altitude < 3.0:
            reward -= 30.0 * (3.0 - altitude) ** 2
        
        if vertical_vel < -1.5:
            reward -= 15.0 * (abs(vertical_vel) - 1.5)
        
        # Catastrophic penalties
        if tilt > 1.57:
            reward -= 50.0
        
        if angular_vel_mag > 4.0:
            reward -= 30.0
        
        # Action smoothness
        if self.current_step > 0:
            action_change = np.sum(np.abs(self.previous_action - obs[17:21]))
            if action_change > 0.8:
                reward -= 2.0 * action_change
        
        # Survival bonus
        if altitude > 1.0:
            reward += 0.5
        
        return reward
    
    def _check_termination(self, obs):
        """Check episode end"""
        terminated = False
        truncated = False
        info = {}
        
        roll, pitch = obs[12:14]
        tilt = np.sqrt(roll**2 + pitch**2)
        altitude = obs[15]
        vertical_vel = obs[16]
        impact_flag = obs[21]
        
        # SUCCESS: Recovery
        if (impact_flag > 0.5 and 
            self.stable_steps >= self.stable_duration_required and
            self.recovery_time is not None):
            
            terminated = True
            info['termination_reason'] = 'recovery_success'
            info['success'] = True
            self.successful_recoveries += 1
            
            if self.enable_logging:
                print(f"✓ RECOVERY in {self.recovery_time} steps")
        
        # SUCCESS: Hover (stage 1 only)
        elif (self.training_stage == TrainingStage.HOVER and 
              self.stable_steps >= self.stable_duration_required):
            
            terminated = True
            info['termination_reason'] = 'hover_success'
            info['success'] = True
            self.successful_recoveries += 1
            
            if self.enable_logging:
                print(f"✓ HOVER STABLE for {self.stable_steps} steps")
        
        # FAILURE: Ground
        elif altitude <= self.ground_altitude_limit:
            truncated = True
            info['termination_reason'] = 'ground_crash'
            info['success'] = False
            self.crashes += 1
            
            if self.enable_logging:
                print(f"✗ GROUND CRASH at {altitude:.3f}m")
        
        # FAILURE: Fast descent
        elif altitude < 1.0 and vertical_vel < self.crash_velocity_threshold:
            truncated = True
            info['termination_reason'] = 'crash_velocity'
            info['success'] = False
            self.crashes += 1
        
        # FAILURE: Inverted
        elif tilt > np.pi * 0.9:
            truncated = True
            info['termination_reason'] = 'inverted'
            info['success'] = False
            self.crashes += 1
        
        # FAILURE: Out of bounds
        elif altitude > self.max_altitude:
            truncated = True
            info['termination_reason'] = 'out_of_bounds'
            info['success'] = False
        
        # TIMEOUT
        elif self.current_step >= self.max_episode_steps:
            truncated = True
            info['termination_reason'] = 'max_steps'
            info['success'] = self.recovery_time is not None
        
        # AirSim collision
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            truncated = True
            info['termination_reason'] = 'collision'
            info['success'] = False
            self.crashes += 1
        
        return terminated, truncated, info
    
    def _apply_small_disturbance(self):
        """Small push for disturbance stage"""
        if self.enable_logging:
            print(f"\n💨 Small disturbance")
        
        direction = np.random.choice(['left', 'right', 'front', 'back'])
        force = np.random.uniform(10, 25)
        
        force_map = {
            'left': (0, -force, 0),
            'right': (0, force, 0),
            'front': (force, 0, 0),
            'back': (-force, 0, 0)
        }
        fx, fy, fz = force_map[direction]
        
        self.client.moveByVelocityAsync(fx*0.1, fy*0.1, 0, duration=0.15).join()
        
        angular_vel = np.random.uniform(20, 50)
        roll_rate = np.radians(angular_vel) if direction in ['left', 'right'] else 0
        pitch_rate = np.radians(angular_vel) if direction in ['front', 'back'] else 0
        
        self.client.moveByAngleRatesThrottleAsync(roll_rate, pitch_rate, 0, 0.59, duration=0.2).join()
        
        self.impact_occurred = True
        self.steps_since_impact = 0
    
    def _apply_impact(self):
        """Full impact - simplified for now"""
        if self.enable_logging:
            print(f"\n💥 Bird strike impact")
        
        direction = np.random.choice(['left', 'right', 'front', 'back'])
        force = np.random.uniform(60, 120)
        
        force_map = {
            'left': (0, -force, 0), 'right': (0, force, 0),
            'front': (force, 0, 0), 'back': (-force, 0, 0)
        }
        fx, fy, fz = force_map[direction]
        
        self.client.moveByVelocityAsync(fx*0.15, fy*0.15, 0, duration=0.2).join()
        
        angular_vel = np.random.uniform(40, 100)
        roll_rate = np.radians(angular_vel) if direction in ['left', 'right'] else 0
        pitch_rate = np.radians(angular_vel) if direction in ['front', 'back'] else 0
        
        self.client.moveByAngleRatesThrottleAsync(roll_rate, pitch_rate, 0, 0.59, duration=0.35).join()
        
        self.impact_occurred = True
        self.steps_since_impact = 0
    
    def _get_info(self):
        """Episode info"""
        success_rate = (
            self.successful_recoveries / max(1, self.total_episodes) * 100
        )
        
        return {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'training_stage': self.training_stage.value,
            'impact_occurred': self.impact_occurred,
            'steps_since_impact': self.steps_since_impact,
            'recovery_time': self.recovery_time,
            'stable_steps': self.stable_steps,
            'peak_tilt': self.peak_tilt,
            'peak_angular_vel': self.peak_angular_vel,
            'lowest_altitude': self.lowest_altitude,
            'altitude_loss': self.spawn_altitude - self.lowest_altitude,
            'total_episodes': self.total_episodes,
            'successful_recoveries': self.successful_recoveries,
            'crashes': self.crashes,
            'success_rate': success_rate
        }
    
    def close(self):
        """Cleanup"""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.client.reset()


def get_baseline_action(obs):
    """
    Simple baseline controller
    """
    roll = obs[12]
    pitch = obs[13]
    
    # Start at 0.5 (will transform to 0.66 hover)
    action = np.array([0.5, 0.5, 0.5, 0.5])
    
    kp = 0.4
    
    # Corrections
    action[2] += kp * roll  # FL
    action[1] += kp * roll  # RL
    action[0] -= kp * roll  # FR
    action[3] -= kp * roll  # RR
    
    action[1] += kp * pitch  # RL
    action[3] += kp * pitch  # RR
    action[0] -= kp * pitch  # FR
    action[2] -= kp * pitch  # FL
    
    return np.clip(action, 0, 1)


if __name__ == "__main__":
    print("="*70)
    print("🧪 TESTING CONTINUOUS CONTROL")
    print("="*70)
    
    env = PostImpactRecoveryEnv(
        training_stage="hover",
        enable_logging=True
    )
    
    obs, info = env.reset()
    
    for step in range(200):
        action = get_baseline_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 50 == 0:
            tilt = np.sqrt(obs[12]**2 + obs[13]**2)
            alt = obs[15]
            print(f"Step {step}: Alt={alt:.2f}m, Tilt={np.degrees(tilt):.1f}°, Reward={reward:.2f}")
        
        if terminated or truncated:
            print(f"\nEnded: {info.get('termination_reason')}")
            break
    
    env.close()
    print("\n✅ Test complete!")