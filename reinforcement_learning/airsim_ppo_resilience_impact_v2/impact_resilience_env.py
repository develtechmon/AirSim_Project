"""
Impact Resilience Environment - FINAL WORKING VERSION
======================================================

THIS IS THE CORRECT VERSION THAT ALLOWS LEARNING!

Key fixes:
1. Impact does NOT end episode immediately
2. Collision detection disabled during impact phase
3. Drone has 50+ steps to learn recovery
4. Only terminates on ground collision or timeout
5. Gives clear learning signal (reward for recovery)

How it works:
- Steps 1-29: Normal hover
- Step 30: IMPACT applied
- Steps 31-80: Drone tries to recover (LEARNING PHASE)
- Episode ends only if: ground hit OR timeout OR successful recovery
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from impact_simulator import ImpactSimulator, ImpactType
from feature_extraction import IMUFeatureExtractor, IMUBuffer


class ImpactResilienceEnv(gym.Env):
    """
    FINAL WORKING VERSION - Allows drone to learn recovery from impacts.
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
        self.feature_extractor = IMUFeatureExtractor(sampling_rate=10.0)
        self.imu_buffer = IMUBuffer(maxlen=50)
        
        # Hover target
        self.hover_target = np.array([0.0, 0.0, -5.0])  # 5m altitude
        
        # Episode settings
        self.steps = 0
        self.max_steps = 200  # 20 seconds total
        self.impact_step = 30  # Apply impact at step 30
        self.min_recovery_steps = 50  # Give at least 50 steps to recover
        
        # Episode state
        self.impact_applied = False
        self.impact_time = 0
        self.recovered = False
        self.steps_since_impact = 0
        self.recovery_start_time = None
        
        # Collision tracking
        self.collision_grace_period = 5  # Ignore collisions for 5 steps after impact
        self.last_collision_check = 0
        
        # Action space: [vx, vy, vz, yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0, -np.pi/2]),
            high=np.array([5.0, 5.0, 5.0, np.pi/2]),
            dtype=np.float32
        )
        
        # Observation space: 42 values
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(42,),
            dtype=np.float32
        )
        
        print("🚁 Impact Resilience Environment - FINAL VERSION")
        print(f"   Impact at step: {self.impact_step}")
        print(f"   Min recovery time: {self.min_recovery_steps} steps")
        print(f"   Max episode: {self.max_steps} steps")
    
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
        self.recovered = False
        self.steps_since_impact = 0
        self.recovery_start_time = None
        self.last_collision_check = 0
        
        # Clear buffers
        self.imu_buffer.clear()
        self.impact_simulator.clear_impact()
        
        # Initialize IMU buffer
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
        except:
            pass
    
    def _get_observation(self):
        """Build observation vector (42 values)."""
        
        self._update_imu_buffer()
        
        # Extract IMU features (27 values)
        imu_features = self.feature_extractor.extract_features(self.imu_buffer)
        
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
        
        current_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        current_vel = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
        
        pos_error = current_pos - self.hover_target
        
        if len(self.imu_buffer.gyro) > 0:
            angular_vel = self.imu_buffer.gyro[-1]
        else:
            angular_vel = np.array([0, 0, 0], dtype=np.float32)
        
        altitude = -current_pos[2]
        speed = np.linalg.norm(current_vel)
        angular_speed = np.linalg.norm(angular_vel)
        
        time_since_impact = 0.0
        if self.impact_applied:
            time_since_impact = self.steps_since_impact / 100.0
        
        # Drone state (15 values)
        drone_state = np.array([
            pos_error[0], pos_error[1], pos_error[2],
            current_vel[0], current_vel[1], current_vel[2],
            angular_vel[0], angular_vel[1], angular_vel[2],
            altitude,
            speed,
            angular_speed,
            time_since_impact,
            float(self.impact_applied),
            float(self.recovered),
        ], dtype=np.float32)
        
        obs = np.concatenate([imu_feature_array, drone_state]).astype(np.float32)
        
        return obs
    
    def step(self, action):
        """Execute one environment step."""
        self.steps += 1
        
        # Apply impact at fixed step (GUARANTEED)
        if not self.impact_applied and self.steps == self.impact_step:
            self._apply_impact()
            self.impact_applied = True
            self.impact_time = self.steps
            self.recovery_start_time = time.time()
        
        # Track time since impact
        if self.impact_applied:
            self.steps_since_impact += 1
        
        # Update impact simulator
        self.impact_simulator.update()
        
        # Execute action
        self._execute_action(action)
        
        # Wait for physics
        time.sleep(0.1)
        
        # Get new state
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        new_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        vel = state.kinematics_estimated.linear_velocity
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        speed = np.linalg.norm(velocity)
        
        if len(self.imu_buffer.gyro) > 0:
            angular_vel = self.imu_buffer.gyro[-1]
            angular_speed = np.linalg.norm(angular_vel)
        else:
            angular_speed = 0.0
        
        altitude = -new_pos[2]
        
        # Compute reward
        reward = self._compute_reward(new_pos, velocity, speed, angular_speed, altitude)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # CRITICAL FIX: Only check collisions AFTER grace period
        if self.impact_applied and self.steps_since_impact > self.collision_grace_period:
            
            # Ground collision check (altitude-based - more reliable)
            if altitude < 0.3:  # Very close to ground
                terminated = True
                reward -= 500.0
                print(f"💥 GROUND COLLISION at step {self.steps} (alt={altitude:.2f}m)")
            
            # Wall/object collision (AirSim detection)
            elif self.steps - self.last_collision_check > 3:  # Check every 3 steps
                collision_info = self.client.simGetCollisionInfo()
                self.last_collision_check = self.steps
                
                if collision_info.has_collided:
                    # Verify it's a real collision (not just impact effect)
                    if collision_info.penetration_depth > 0.1:  # Real collision
                        terminated = True
                        reward -= 500.0
                        print(f"💥 OBJECT COLLISION at step {self.steps}")
        
        # Timeout check
        if self.steps >= self.max_steps:
            truncated = True
            if not self.recovered:
                reward -= 200.0
                print(f"⏱️  TIMEOUT - Failed to recover")
        
        # Success check: Recovery achieved
        pos_error = np.linalg.norm(new_pos - self.hover_target)
        
        if self.impact_applied and not self.recovered:
            # Must wait minimum recovery time AND be stable
            if self.steps_since_impact > self.min_recovery_steps:
                if pos_error < 1.5 and speed < 1.0 and angular_speed < 0.4:
                    self.recovered = True
                    recovery_time = self.steps_since_impact
                    reward += 1000.0  # HUGE reward for successful recovery
                    print(f"✅ RECOVERY SUCCESS in {recovery_time} steps!")
                    # Don't terminate - let it continue to max_steps for more learning
        
        # Info dict
        info = {
            'position_error': float(pos_error),
            'speed': float(speed),
            'angular_speed': float(angular_speed),
            'altitude': float(altitude),
            'impact_applied': self.impact_applied,
            'recovered': self.recovered,
            'collision': terminated and not truncated,
            'steps': self.steps,
            'steps_since_impact': self.steps_since_impact,
            'recovery_time': self.steps_since_impact if self.recovered else 0,
            'impact_type': getattr(self, 'current_impact_type', 'none')
        }
        
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def _apply_impact(self):
        """
        Apply impact to drone.
        
        Uses moderate magnitude for learning.
        """
        # Random impact type
        impact_type = np.random.choice(list(ImpactType))
        
        # Moderate magnitude (1.0-2.0 range for learning)
        magnitude = np.random.uniform(1.0, 2.0)
        
        # Random horizontal direction (avoid straight down)
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        direction = direction / np.linalg.norm(direction)
        
        # Apply impact
        params = self.impact_simulator.apply_impact(
            impact_type=impact_type,
            magnitude=magnitude,
            direction=direction
        )
        
        self.current_impact_type = params.impact_type.value
        
        print(f"\n{'='*70}")
        print(f"💥 IMPACT APPLIED at step {self.steps}")
        print(f"{'='*70}")
        print(f"  Type:      {params.impact_type.value}")
        print(f"  Magnitude: {params.magnitude:.2f}")
        print(f"  Direction: [{params.direction[0]:.2f}, {params.direction[1]:.2f}, {params.direction[2]:.2f}]")
        print(f"  Duration:  {params.duration:.2f}s")
        print(f"  Agent has {self.min_recovery_steps} steps minimum to recover...")
        print(f"{'='*70}\n")
    
    def _execute_action(self, action):
        """Execute recovery action."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        try:
            self.client.moveByVelocityAsync(
                vx=float(action[0]),
                vy=float(action[1]),
                vz=float(action[2]),
                duration=0.1,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(np.degrees(action[3])))
            )
        except:
            pass
    
    def _compute_reward(self, position, velocity, speed, angular_speed, altitude):
        """
        Reward function for learning recovery.
        
        KEY: More reward during recovery phase to guide learning.
        """
        reward = 0.0
        
        pos_error = np.linalg.norm(position - self.hover_target)
        
        if not self.impact_applied:
            # Before impact: Small reward for hovering
            if pos_error < 1.0 and speed < 0.5:
                reward += 2.0
            reward -= pos_error * 0.5
        
        else:
            # After impact: STRONG rewards for recovery behavior
            
            # 1. ALTITUDE - MOST CRITICAL (avoid ground)
            if altitude > 4.0:
                reward += 100.0  # Excellent altitude
            elif altitude > 3.0:
                reward += 50.0   # Good altitude
            elif altitude > 2.0:
                reward += 20.0   # Acceptable
            elif altitude > 1.0:
                reward += 5.0    # Danger zone
            else:
                reward -= 200.0  # Critical danger!
            
            # 2. POSITION - Getting back to hover point
            if pos_error < 3.0:
                reward += 40.0
            if pos_error < 2.0:
                reward += 60.0
            if pos_error < 1.0:
                reward += 80.0
            reward -= pos_error * 10.0
            
            # 3. VELOCITY - Damping oscillations
            if speed < 2.0:
                reward += 30.0
            if speed < 1.0:
                reward += 50.0
            if speed < 0.5:
                reward += 70.0
            reward -= speed * 15.0
            
            # 4. ANGULAR VELOCITY - Stop spinning
            if angular_speed < 1.0:
                reward += 30.0
            if angular_speed < 0.5:
                reward += 50.0
            if angular_speed < 0.2:
                reward += 70.0
            reward -= angular_speed * 20.0
            
            # 5. SURVIVAL BONUS - Reward for each step survived
            if self.steps_since_impact > 10:
                reward += 5.0
            if self.steps_since_impact > 30:
                reward += 10.0
            if self.steps_since_impact > 50:
                reward += 15.0
            
            # 6. STABILITY BONUS - All metrics good simultaneously
            if pos_error < 2.0 and speed < 1.0 and angular_speed < 0.5 and altitude > 3.0:
                reward += 50.0  # Good overall state
        
        # Small time penalty
        reward -= 1.0
        
        return reward
    
    def close(self):
        """Clean up."""
        try:
            self.impact_simulator.clear_impact()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass
    
    def render(self):
        pass


# Test
if __name__ == "__main__":
    print("="*70)
    print("IMPACT RESILIENCE ENVIRONMENT - FINAL TEST")
    print("="*70)
    
    env = ImpactResilienceEnv()
    
    print("\nTesting 2 episodes with random actions...")
    print("Watch for: Impact → Struggle → Recovery or Fail\n")
    
    for ep in range(2):
        print(f"\n{'='*70}")
        print(f"EPISODE {ep+1}")
        print(f"{'='*70}")
        
        obs, info = env.reset()
        
        for step in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print key milestones
            if step == 30:
                print(f"Step {step}: Impact should be applied...")
            
            if info['impact_applied'] and step > 30 and step % 15 == 0:
                print(f"Step {step}: Recovering... PosErr={info['position_error']:.2f}m, Alt={info['altitude']:.2f}m")
            
            if terminated or truncated:
                print(f"\n{'='*70}")
                print(f"Episode ended at step {step}")
                print(f"  Impact applied: {info['impact_applied']}")
                print(f"  Recovered: {info['recovered']}")
                print(f"  Steps after impact: {info['steps_since_impact']}")
                print(f"{'='*70}")
                break
    
    env.close()
    print("\n✅ Test complete! Ready for training.")