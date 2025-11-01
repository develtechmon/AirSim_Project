import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import time
import random

class DroneFastRecoveryEnv(gym.Env):
    """
    Fast Mid-Air Recovery Environment
    
    Goal: Recover from violent disturbances BEFORE hitting ground
    - Drone must stabilize quickly after crazy flips
    - Must return to hover at same altitude
    - Penalized heavily for altitude loss
    - Rewarded for fast recovery time
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 ip_address="127.0.0.1",
                 step_length=0.1):  # Faster control loop!
        
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        self.step_length = step_length
        
        # Training parameters
        self.max_episode_steps = 300  # 30 seconds at 0.1s steps
        self.current_step = 0
        
        # CRITICAL: High starting altitude for recovery time
        self.nominal_altitude = -25.0  # 25 meters high
        self.target_pos = np.array([0.0, 0.0, self.nominal_altitude])
        
        # Recovery criteria
        self.altitude_tolerance = 3.0  # ±3m from nominal
        self.tilt_tolerance = 0.26  # ~15 degrees
        self.velocity_tolerance = 2.0  # m/s
        
        # Ground crash altitude
        self.ground_level = -2.0  # Crash if below 2m
        
        # Observation space: 21 dimensions
        # [pos_x, pos_y, pos_z, 
        #  roll, pitch, yaw,
        #  vel_x, vel_y, vel_z, 
        #  ang_vel_x, ang_vel_y, ang_vel_z,
        #  altitude_error, tilt_angle, is_stable,
        #  time_since_disturbance, falling_speed,
        #  recovery_urgency, altitude_loss, ground_proximity, time_to_ground]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),
            dtype=np.float32
        )
        
        # Action space: [roll_rate, pitch_rate, yaw_rate, throttle]
        # More aggressive range for fast recovery
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Disturbance tracking
        self.disturbance_applied = False
        self.disturbance_step = 0
        self.initial_altitude_before_disturbance = self.nominal_altitude
        self.is_in_recovery_mode = False
        self.recovery_start_step = 0
        
        # Episode statistics
        self.episode_stats = {
            'total_reward': 0,
            'successful_recoveries': 0,
            'crashes': 0,
            'recovery_time': 0,
            'max_tilt': 0,
            'altitude_loss': 0,
            'closest_to_ground': 100
        }
        
        print("✅ Fast Recovery Environment initialized")
        print(f"   Nominal altitude: {-self.nominal_altitude}m")
        print(f"   Control frequency: {1/step_length}Hz")
        print(f"   Recovery window: ~{(self.nominal_altitude - self.ground_level):.1f}m")
    
    def reset(self, seed=None, options=None):
        """Reset to starting conditions"""
        super().reset(seed=seed)
        
        # Reset drone
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Take off to HIGH altitude for recovery testing
        self.client.takeoffAsync().join()
        self.client.moveToPositionAsync(
            self.target_pos[0],
            self.target_pos[1],
            self.target_pos[2],
            velocity=5
        ).join()
        
        time.sleep(1.5)
        
        # Reset tracking variables
        self.current_step = 0
        self.disturbance_applied = False
        self.disturbance_step = 0
        self.initial_altitude_before_disturbance = self.nominal_altitude
        self.is_in_recovery_mode = False
        self.recovery_start_step = 0
        
        self.episode_stats = {
            'total_reward': 0,
            'successful_recoveries': 0,
            'crashes': 0,
            'recovery_time': 0,
            'max_tilt': 0,
            'altitude_loss': 0,
            'closest_to_ground': 100
        }
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Execute one control step"""
        self.current_step += 1
        
        # Denormalize actions - AGGRESSIVE for fast recovery
        roll_rate = action[0] * 15.0  # ±15 rad/s (860°/s!)
        pitch_rate = action[1] * 15.0
        yaw_rate = action[2] * 8.0
        throttle = np.clip((action[3] + 1) * 0.5, 0.0, 1.0)  # 0 to 1
        
        # Apply action
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=float(roll_rate),
            pitch_rate=float(pitch_rate),
            yaw_rate=float(yaw_rate),
            throttle=float(throttle),
            duration=self.step_length
        )
        
        time.sleep(self.step_length)
        
        # Apply disturbance at specific time
        if not self.disturbance_applied and self.current_step == 10:
            self._apply_violent_disturbance()
        
        # Get observation
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_episode_steps
        info = self._get_info()
        
        self.episode_stats['total_reward'] += reward
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Get observation with recovery-specific features"""
        state = self.client.getMultirotorState()
        
        # Position
        pos = state.kinematics_estimated.position
        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        # Orientation
        orientation = state.kinematics_estimated.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation)
        
        # Velocities
        vel = state.kinematics_estimated.linear_velocity
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        
        ang_vel = state.kinematics_estimated.angular_velocity
        angular_velocity = np.array([ang_vel.x_val, ang_vel.y_val, ang_vel.z_val])
        
        # Recovery-specific features
        altitude_error = position[2] - self.nominal_altitude
        tilt = np.sqrt(roll**2 + pitch**2)
        
        # Check if stable
        is_stable = (
            abs(altitude_error) < self.altitude_tolerance and
            tilt < self.tilt_tolerance and
            np.linalg.norm(velocity) < self.velocity_tolerance
        )
        
        # Time since disturbance
        time_since_disturbance = max(0, self.current_step - self.disturbance_step)
        
        # Falling speed (positive = falling down)
        falling_speed = velocity[2]  # NED: +Z = down
        
        # Recovery urgency (0-1, higher = more urgent)
        ground_distance = abs(position[2] - self.ground_level)
        recovery_urgency = 1.0 - min(ground_distance / 20.0, 1.0)
        
        # Altitude loss since disturbance
        altitude_loss = self.initial_altitude_before_disturbance - position[2]
        
        # Ground proximity (inverse of distance)
        ground_proximity = 1.0 / max(ground_distance, 0.1)
        
        # Time to ground (estimate)
        if falling_speed > 0.1:
            time_to_ground = ground_distance / falling_speed
        else:
            time_to_ground = 100.0  # Not falling
        
        # Track stats
        if tilt > self.episode_stats['max_tilt']:
            self.episode_stats['max_tilt'] = tilt
        
        if ground_distance < self.episode_stats['closest_to_ground']:
            self.episode_stats['closest_to_ground'] = ground_distance
        
        # Construct observation
        obs = np.array([
            position[0], position[1], position[2],
            roll, pitch, yaw,
            velocity[0], velocity[1], velocity[2],
            angular_velocity[0], angular_velocity[1], angular_velocity[2],
            altitude_error,
            tilt,
            float(is_stable),
            time_since_disturbance * 0.1,  # Normalize
            falling_speed,
            recovery_urgency,
            altitude_loss,
            ground_proximity,
            min(time_to_ground / 10.0, 10.0)  # Normalize
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self):
        """Reward function optimized for FAST recovery"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        orientation = state.kinematics_estimated.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation)
        tilt = np.sqrt(roll**2 + pitch**2)
        
        vel = state.kinematics_estimated.linear_velocity
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        
        reward = 0.0
        
        # === CRITICAL: Survival (staying above ground) ===
        ground_distance = abs(position[2] - self.ground_level)
        
        if ground_distance < 3.0:
            # EMERGENCY: Very close to ground!
            reward -= 50.0 * (1.0 - ground_distance / 3.0)
        
        # === Altitude maintenance (VERY important) ===
        altitude_error = abs(position[2] - self.nominal_altitude)
        
        if altitude_error < 2.0:
            reward += 5.0  # Good altitude
        elif altitude_error < 5.0:
            reward += 2.0  # Acceptable
        else:
            reward -= altitude_error * 2.0  # Bad - losing altitude
        
        # === Tilt penalty (must level out) ===
        if tilt < 0.26:  # < 15 degrees
            reward += 3.0
        elif tilt < 0.52:  # < 30 degrees
            reward += 1.0
        elif tilt < 1.0:  # < 57 degrees
            reward -= 2.0
        else:  # Severely tilted or upside down
            reward -= 10.0
        
        # === Velocity stabilization ===
        speed = np.linalg.norm(velocity)
        if speed < 1.0:
            reward += 2.0
        elif speed < 3.0:
            reward += 0.5
        else:
            reward -= speed * 0.5
        
        # === Recovery bonus (BIG reward for successful recovery) ===
        if self.is_in_recovery_mode:
            is_stable = (
                altitude_error < self.altitude_tolerance and
                tilt < self.tilt_tolerance and
                speed < self.velocity_tolerance
            )
            
            if is_stable:
                recovery_time = self.current_step - self.recovery_start_step
                
                # HUGE bonus for fast recovery
                time_bonus = max(50.0 - recovery_time, 10.0)
                reward += time_bonus
                
                self.is_in_recovery_mode = False
                self.episode_stats['successful_recoveries'] += 1
                self.episode_stats['recovery_time'] = recovery_time
                
                print(f"✅ RECOVERY SUCCESS in {recovery_time} steps ({recovery_time*self.step_length:.1f}s)!")
                print(f"   Bonus: +{time_bonus:.1f}")
        
        # === Time penalty during recovery (encourages speed) ===
        if self.disturbance_applied and not self.is_in_recovery_mode:
            reward -= 0.5  # Penalty for each step not recovered
        
        # === Falling penalty ===
        falling_speed = velocity[2]
        if falling_speed > 3.0:  # Falling fast
            reward -= falling_speed * 2.0
        
        return float(reward)
    
    def _check_terminated(self):
        """Check termination - CRASH only"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        # CRASHED - hit ground
        if pos.z_val > self.ground_level:
            self.episode_stats['crashes'] += 1
            altitude_lost = self.initial_altitude_before_disturbance - pos.z_val
            print(f"💥 CRASHED! Lost {altitude_lost:.1f}m altitude")
            print(f"   Max tilt: {np.degrees(self.episode_stats['max_tilt']):.1f}°")
            return True
        
        # Flew away (failed to recover position)
        horizontal_dist = np.sqrt(pos.x_val**2 + pos.y_val**2)
        if horizontal_dist > 50:
            print(f"🚫 FLEW AWAY - {horizontal_dist:.1f}m from origin")
            return True
        
        # Collision
        collision = self.client.simGetCollisionInfo()
        if collision.has_collided:
            self.episode_stats['crashes'] += 1
            print(f"💥 COLLISION!")
            return True
        
        return False
    
    def _apply_violent_disturbance(self):
        """Apply VIOLENT disturbance - crazy flip scenario"""
        print(f"\n{'='*60}")
        print(f"⚡⚡⚡ VIOLENT DISTURBANCE at step {self.current_step} ⚡⚡⚡")
        
        # Save pre-disturbance altitude
        state = self.client.getMultirotorState()
        self.initial_altitude_before_disturbance = state.kinematics_estimated.position.z_val
        
        # Choose random disturbance type
        disturbance_type = random.choice([
            'eagle_strike',
            'extreme_tumble',
            'propeller_failure',
            'wind_shear'
        ])
        
        print(f"   Type: {disturbance_type}")
        
        if disturbance_type == 'eagle_strike':
            # Massive bird strike - violent impact
            magnitude = 80
            angle = random.uniform(0, 2*np.pi)
            
            wind = airsim.Vector3r(
                magnitude * np.cos(angle),
                magnitude * np.sin(angle),
                -15
            )
            self.client.simSetWind(wind)
            
            # EXTREME tumble
            self.client.moveByAngleRatesThrottleAsync(
                roll_rate=random.uniform(-18, 18),
                pitch_rate=random.uniform(-18, 18),
                yaw_rate=random.uniform(-12, 12),
                throttle=0.1,  # Almost no thrust
                duration=0.8
            ).join()
            
            # Continue chaos
            self.client.moveByAngleRatesThrottleAsync(
                roll_rate=random.uniform(-15, 15),
                pitch_rate=random.uniform(-15, 15),
                yaw_rate=random.uniform(-10, 10),
                throttle=0.2,
                duration=0.6
            ).join()
            
            # Gradual wind reduction
            time.sleep(0.6)
            for i in range(3):
                factor = 1.0 - (i * 0.3)
                self.client.simSetWind(airsim.Vector3r(
                    wind.x_val * factor,
                    wind.y_val * factor,
                    wind.z_val * factor
                ))
                time.sleep(0.2)
            
            self.client.simSetWind(airsim.Vector3r(0, 0, 0))
            
        elif disturbance_type == 'extreme_tumble':
            # Pure violent tumbling
            wind = airsim.Vector3r(
                random.uniform(-50, 50),
                random.uniform(-50, 50),
                random.uniform(-20, 10)
            )
            self.client.simSetWind(wind)
            
            for _ in range(3):
                self.client.moveByAngleRatesThrottleAsync(
                    roll_rate=random.uniform(-20, 20),
                    pitch_rate=random.uniform(-20, 20),
                    yaw_rate=random.uniform(-15, 15),
                    throttle=random.uniform(0.1, 0.3),
                    duration=0.4
                )
                time.sleep(0.4)
            
            self.client.simSetWind(airsim.Vector3r(0, 0, 0))
            
        elif disturbance_type == 'propeller_failure':
            # Asymmetric thrust - uncontrollable spin
            wind = airsim.Vector3r(
                random.uniform(-40, 40),
                random.uniform(-40, 40),
                -15
            )
            self.client.simSetWind(wind)
            
            # Massive yaw with falling
            for _ in range(4):
                self.client.moveByAngleRatesThrottleAsync(
                    roll_rate=random.uniform(-8, 8),
                    pitch_rate=random.uniform(-8, 8),
                    yaw_rate=random.uniform(-25, 25),  # INSANE yaw
                    throttle=0.2,
                    duration=0.3
                )
                time.sleep(0.3)
            
            self.client.simSetWind(airsim.Vector3r(0, 0, 0))
            
        else:  # wind_shear
            # Sudden extreme wind from different directions
            for _ in range(3):
                wind = airsim.Vector3r(
                    random.uniform(-60, 60),
                    random.uniform(-60, 60),
                    random.uniform(-15, 15)
                )
                self.client.simSetWind(wind)
                
                self.client.moveByAngleRatesThrottleAsync(
                    roll_rate=random.uniform(-12, 12),
                    pitch_rate=random.uniform(-12, 12),
                    yaw_rate=random.uniform(-8, 8),
                    throttle=0.25,
                    duration=0.4
                )
                time.sleep(0.4)
            
            self.client.simSetWind(airsim.Vector3r(0, 0, 0))
        
        self.disturbance_applied = True
        self.disturbance_step = self.current_step
        self.is_in_recovery_mode = True
        self.recovery_start_step = self.current_step + 1
        
        print(f"{'='*60}\n")
    
    def _get_info(self):
        """Get episode info"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        return {
            'episode_step': self.current_step,
            'disturbance_applied': self.disturbance_applied,
            'is_recovering': self.is_in_recovery_mode,
            'current_altitude': -pos.z_val,
            'ground_distance': abs(pos.z_val - self.ground_level),
            'episode_stats': self.episode_stats.copy()
        }
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)