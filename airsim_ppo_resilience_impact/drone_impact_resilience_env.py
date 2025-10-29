"""
Impact Resilience Environment for PPO Training
===============================================

Complete environment for training impact-resilient drones using PPO.

Components:
1. Impact Simulator - generates different impact types
2. Feature Extractor - extracts IMU signatures
3. Gym Environment - standard RL interface
4. Recovery Controller - learned via PPO

Training Objective:
The agent learns to:
1. Detect impacts from IMU data
2. Classify impact type from features
3. Execute appropriate recovery strategy
4. Stabilize and return to hover

Reward Structure:
+500  = Successful recovery (back to stable hover)
+100  = Altitude maintained during recovery
+50   = Angular velocity reduced (stop spinning)
+30   = Position error reduced
-100  = Ground collision
-200  = Failed to recover (timeout)
-1    = Time penalty (encourage fast recovery)
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from collections import deque
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, '/home/claude')

# Import our custom modules
from impact_simulator import ImpactSimulator, ImpactType
from feature_extraction import IMUFeatureExtractor, IMUBuffer


class ImpactResilienceEnv(gym.Env):
    """
    Gymnasium environment for training impact-resilient drone control.
    
    The agent observes:
    - IMU features (27 values from feature extractor)
    - Drone state (15 values: position, velocity, etc.)
    Total: 42 observation dimensions
    
    The agent outputs:
    - Recovery control commands (4 actions: vx, vy, vz, yaw_rate)
    
    Analogy: Like teaching someone to recover from being pushed -
    they need to sense how they're falling (IMU), then react
    appropriately (thrust/orientation corrections).
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # AirSim connection
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Components
        self.impact_simulator = ImpactSimulator(self.client)
        self.feature_extractor = IMUFeatureExtractor(sampling_rate=10.0)  # 10Hz
        self.imu_buffer = IMUBuffer(maxlen=50)  # 5 seconds of data
        
        # Hover target (recovery goal)
        self.hover_target = np.array([0.0, 0.0, -5.0])  # 5m altitude
        
        # Episode tracking
        self.steps = 0
        self.max_steps = 300  # 30 seconds at 10Hz
        self.impact_applied = False
        self.impact_time = 0
        self.recovery_start_time = None
        self.recovered = False
        
        # Statistics
        self.episode_impacts = []
        self.episode_recovery_times = []
        
        # Action space: [vx, vy, vz, yaw_rate]
        # Agent controls velocity corrections for recovery
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0, -np.pi/2]),
            high=np.array([5.0, 5.0, 5.0, np.pi/2]),
            dtype=np.float32
        )
        
        # Observation space: IMU features (27) + drone state (15) = 42 values
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(42,),
            dtype=np.float32
        )
        
        print("🚁 Impact Resilience Environment initialized")
        print(f"   Max steps: {self.max_steps}")
        print(f"   Observation dim: {self.observation_space.shape[0]}")
        print(f"   Action dim: {self.action_space.shape[0]}")
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset AirSim
        self.client.reset()
        time.sleep(0.1)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Takeoff
        self.client.takeoffAsync().join()
        time.sleep(0.3)
        
        # Move to hover position
        self.client.moveToPositionAsync(
            float(self.hover_target[0]),
            float(self.hover_target[1]),
            float(self.hover_target[2]),
            velocity=3.0
        ).join()
        time.sleep(0.5)
        
        # Reset state
        self.steps = 0
        self.impact_applied = False
        self.impact_time = 0
        self.recovery_start_time = None
        self.recovered = False
        
        # Clear buffers
        self.imu_buffer.clear()
        self.impact_simulator.clear_impact()
        
        # Initialize IMU buffer with normal flight data
        for _ in range(50):
            self._update_imu_buffer()
            time.sleep(0.01)
        
        return self._get_observation(), {}
    
    def _update_imu_buffer(self):
        """Read IMU and update buffer"""
        try:
            imu_data = self.client.getImuData()
            
            accel = np.array([
                imu_data.linear_acceleration.x_val,
                imu_data.linear_acceleration.y_val,
                imu_data.linear_acceleration.z_val
            ])
            
            gyro = np.array([
                imu_data.angular_velocity.x_val,
                imu_data.angular_velocity.y_val,
                imu_data.angular_velocity.z_val
            ])
            
            self.imu_buffer.add_sample(accel, gyro, time.time())
        except Exception as e:
            print(f"Warning: IMU read failed: {e}")
    
    def _get_observation(self):
        """
        Build observation vector (42 values total).
        
        Structure:
        - IMU features [0-26]: 27 features from feature extractor
        - Drone state [27-41]: 15 state variables
        """
        # Update IMU buffer
        self._update_imu_buffer()
        
        # Extract IMU features (27 values)
        imu_features = self.feature_extractor.extract_features(self.imu_buffer)
        
        # Convert feature dict to array (fixed order)
        feature_keys = [
            'jerk_max', 'jerk_mean', 'jerk_std', 'jerk_x', 'jerk_y', 'jerk_z',
            'jerk_dominant_axis', 'fft_dominant_freq', 'fft_total_power',
            'fft_low_freq_ratio', 'fft_high_freq_ratio', 'fft_spectral_centroid',
            'fft_gyro_dominant_freq', 'wpd_energy_max', 'wpd_energy_mean',
            'wpd_energy_std', 'wpd_entropy', 'stat_accel_anomaly',
            'stat_accel_mean', 'stat_accel_std', 'stat_accel_range',
            'stat_gyro_mean', 'stat_gyro_std', 'stat_gyro_max',
            'temporal_sustained_count', 'temporal_max_angular_accel',
            'temporal_vertical_accel'
        ]
        
        imu_feature_array = np.array([
            imu_features.get(key, 0.0) for key in feature_keys
        ], dtype=np.float32)
        
        # Get drone state
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        
        # Current state
        current_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        current_vel = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
        
        # Position error from hover target
        pos_error = current_pos - self.hover_target
        
        # Angular velocity from IMU
        if len(self.imu_buffer.gyro) > 0:
            angular_vel = self.imu_buffer.gyro[-1]
        else:
            angular_vel = np.array([0, 0, 0], dtype=np.float32)
        
        # Altitude
        altitude = -current_pos[2]
        
        # Time since impact (normalized)
        time_since_impact = 0.0
        if self.impact_applied:
            time_since_impact = (self.steps - self.impact_time) / self.max_steps
        
        # Speed
        speed = np.linalg.norm(current_vel)
        
        # Angular speed
        angular_speed = np.linalg.norm(angular_vel)
        
        # Drone state (15 values)
        drone_state = np.array([
            pos_error[0], pos_error[1], pos_error[2],  # 3: position error
            current_vel[0], current_vel[1], current_vel[2],  # 3: velocity
            angular_vel[0], angular_vel[1], angular_vel[2],  # 3: angular velocity
            altitude,  # 1: altitude
            speed,  # 1: speed
            angular_speed,  # 1: angular speed
            time_since_impact,  # 1: recovery progress
            float(self.impact_applied),  # 1: impact detected flag
            float(self.recovered),  # 1: recovery status
        ], dtype=np.float32)
        
        # Combine: IMU features (27) + Drone state (15) = 42 total
        obs = np.concatenate([
            imu_feature_array,
            drone_state
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action):
        """Execute one environment step."""
        self.steps += 1
        
        # Apply random impact between steps 20-50 (after stabilization)
        if not self.impact_applied and 20 <= self.steps <= 50:
            # 30% chance per step to apply impact
            if np.random.random() < 0.3:
                self._apply_random_impact()
        
        # Update impact simulator (clears expired impacts)
        self.impact_simulator.update()
        
        # Execute action (recovery control)
        self._execute_action(action)
        
        # Wait for physics (10Hz control rate)
        time.sleep(0.1)
        
        # Get new state
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        new_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        vel = state.kinematics_estimated.linear_velocity
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        speed = np.linalg.norm(velocity)
        
        # Angular velocity
        if len(self.imu_buffer.gyro) > 0:
            angular_vel = self.imu_buffer.gyro[-1]
            angular_speed = np.linalg.norm(angular_vel)
        else:
            angular_speed = 0.0
        
        # Compute reward
        reward = self._compute_reward(new_pos, velocity, speed, angular_speed)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Collision check
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            terminated = True
            reward -= 200.0
            print("💥 COLLISION - Episode terminated")
        
        # Timeout check
        if self.steps >= self.max_steps:
            truncated = True
            if not self.recovered:
                reward -= 200.0  # Failed to recover
                print("⏱️  TIMEOUT - Failed to recover")
        
        # Success check - fully recovered
        pos_error = np.linalg.norm(new_pos - self.hover_target)
        if self.impact_applied and not self.recovered:
            if pos_error < 0.5 and speed < 0.5 and angular_speed < 0.2:
                self.recovered = True
                recovery_time = self.steps - self.impact_time
                self.episode_recovery_times.append(recovery_time)
                reward += 500.0
                print(f"✅ RECOVERY SUCCESS in {recovery_time} steps!")
        
        # Info dict
        info = {
            'position_error': pos_error,
            'speed': speed,
            'angular_speed': angular_speed,
            'impact_applied': self.impact_applied,
            'recovered': self.recovered,
            'steps': self.steps
        }
        
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def _apply_random_impact(self):
        """Apply random impact to drone"""
        params = self.impact_simulator.apply_impact()
        
        self.impact_applied = True
        self.impact_time = self.steps
        self.recovery_start_time = time.time()
        
        self.episode_impacts.append({
            'type': params.impact_type.value,
            'magnitude': params.magnitude,
            'step': self.steps
        })
        
        print(f"\n💥 IMPACT at step {self.steps}:")
        print(f"   Type: {params.impact_type.value}")
        print(f"   Magnitude: {params.magnitude:.2f}")
        print(f"   Direction: {params.direction}")
    
    def _execute_action(self, action):
        """Execute recovery action: [vx, vy, vz, yaw_rate]"""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        self.client.moveByVelocityAsync(
            vx=float(action[0]),
            vy=float(action[1]),
            vz=float(action[2]),
            duration=0.1,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(np.degrees(action[3])))
        )
    
    def _compute_reward(self, position, velocity, speed, angular_speed):
        """Reward function for impact recovery."""
        reward = 0.0
        
        # Position error penalty
        pos_error = np.linalg.norm(position - self.hover_target)
        reward -= pos_error * 2.0
        
        # Altitude maintenance (critical for safety)
        altitude = -position[2]
        altitude_error = abs(altitude - 5.0)
        
        if altitude_error < 0.5:
            reward += 100.0
        elif altitude < 2.0:
            reward -= 50.0
        elif altitude > 10.0:
            reward -= 30.0
        else:
            reward -= altitude_error * 20.0
        
        # Velocity penalty
        if speed < 0.5:
            reward += 50.0
        else:
            reward -= speed * 10.0
        
        # Angular velocity penalty
        if angular_speed < 0.2:
            reward += 50.0
        else:
            reward -= angular_speed * 30.0
        
        # Proximity bonus
        if pos_error < 1.0:
            reward += 30.0
        
        # Stability bonus
        if speed < 0.5 and angular_speed < 0.2:
            reward += 20.0
        
        # Time penalty
        reward -= 1.0
        
        return reward
    
    def close(self):
        """Clean up environment."""
        try:
            self.impact_simulator.clear_impact()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass


# Quick test
if __name__ == "__main__":
    print("="*70)
    print("IMPACT RESILIENCE ENVIRONMENT TEST")
    print("="*70)
    
    env = ImpactResilienceEnv()
    
    print("\nTesting environment reset...")
    obs, info = env.reset()
    print(f"✓ Observation shape: {obs.shape}")
    
    print("\nTesting 50 random steps...")
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            print(f"Step {step}: Reward={reward:.1f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\n✅ Environment test complete!")