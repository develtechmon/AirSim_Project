import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import time
import random
from collections import deque

class DroneRecoveryEnv(gym.Env):
    """
    OpenAI Gym Environment for training drone recovery from disturbances
    
    Goal: Train drone to recover from:
        - Bird strikes (violent tumbling)
        - Wind gusts
        - Collisions
        - Extreme weather
    
    Observation Space: 
        - Position (x, y, z)
        - Orientation (roll, pitch, yaw)
        - Linear velocity (vx, vy, vz)
        - Angular velocity (wx, wy, wz)
        - Target position (x, y, z)
        - Disturbance type indicator
        
    Action Space: 
        - Roll rate, Pitch rate, Yaw rate, Throttle (continuous)
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 ip_address="127.0.0.1",
                 step_length=0.25,
                 image_shape=(84, 84, 1)):
        
        # AirSim setup
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        self.step_length = step_length
        self.image_shape = image_shape
        
        # Training parameters
        self.max_episode_steps = 200
        self.current_step = 0
        
        # Target position (home position)
        self.target_pos = np.array([0.0, 0.0, -10.0])  # 10m altitude
        self.target_tolerance = 2.0  # meters
        
        # Observation space: 18 dimensions
        # [pos_x, pos_y, pos_z, roll, pitch, yaw, 
        #  vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z,
        #  target_x, target_y, target_z, dist_to_target, tilt_angle, disturbance_type]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32
        )
        
        # Action space: [roll_rate, pitch_rate, yaw_rate, throttle]
        # Normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Disturbance types
        self.disturbance_types = {
            0: "none",
            1: "small_bird",
            2: "large_bird",
            3: "wind_gust",
            4: "extreme_wind",
            5: "collision",
            6: "propeller_damage"
        }
        
        self.current_disturbance = 0
        
        # Episode statistics
        self.episode_stats = {
            'total_reward': 0,
            'recoveries': 0,
            'crashes': 0,
            'timeouts': 0,
            'max_tilt': 0,
            'recovery_time': 0
        }
        
        # Recovery tracking
        self.disturbance_applied = False
        self.disturbance_start_time = 0
        self.is_recovering = False
        
        print("✅ Drone Recovery Environment initialized")
    
    def reset(self, seed=None, options=None):
        """Reset environment to starting state"""
        super().reset(seed=seed)
        
        # Reset drone
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Take off to starting position
        self.client.takeoffAsync().join()
        self.client.moveToPositionAsync(
            self.target_pos[0],
            self.target_pos[1],
            self.target_pos[2],
            velocity=3
        ).join()
        
        time.sleep(1)
        
        # Reset episode variables
        self.current_step = 0
        self.disturbance_applied = False
        self.is_recovering = False
        self.current_disturbance = 0
        
        self.episode_stats = {
            'total_reward': 0,
            'recoveries': 0,
            'crashes': 0,
            'timeouts': 0,
            'max_tilt': 0,
            'recovery_time': 0
        }
        
        # Get initial observation
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Execute one time step"""
        self.current_step += 1
        
        # Denormalize actions
        roll_rate = action[0] * 10.0  # -10 to 10 rad/s
        pitch_rate = action[1] * 10.0
        yaw_rate = action[2] * 5.0
        throttle = (action[3] + 1) * 0.5  # 0 to 1
        
        # Apply action
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=float(roll_rate),
            pitch_rate=float(pitch_rate),
            yaw_rate=float(yaw_rate),
            throttle=float(throttle),
            duration=self.step_length
        )
        
        time.sleep(self.step_length)
        
        # Apply disturbance randomly (curriculum learning)
        if not self.disturbance_applied and self.current_step > 5:
            if random.random() < 0.3:  # 30% chance per episode
                self._apply_random_disturbance()
        
        # Get new state
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_episode_steps
        info = self._get_info()
        
        self.episode_stats['total_reward'] += reward
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Get current observation"""
        state = self.client.getMultirotorState()
        
        # Position
        pos = state.kinematics_estimated.position
        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        # Orientation (convert quaternion to euler)
        orientation = state.kinematics_estimated.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation)
        
        # Linear velocity
        vel = state.kinematics_estimated.linear_velocity
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        
        # Angular velocity
        ang_vel = state.kinematics_estimated.angular_velocity
        angular_velocity = np.array([ang_vel.x_val, ang_vel.y_val, ang_vel.z_val])
        
        # Distance to target
        dist_to_target = np.linalg.norm(position - self.target_pos)
        
        # Tilt angle (deviation from upright)
        tilt = np.sqrt(roll**2 + pitch**2)
        
        # Track maximum tilt
        if tilt > self.episode_stats['max_tilt']:
            self.episode_stats['max_tilt'] = tilt
        
        # Construct observation vector
        obs = np.array([
            position[0], position[1], position[2],
            roll, pitch, yaw,
            velocity[0], velocity[1], velocity[2],
            angular_velocity[0], angular_velocity[1], angular_velocity[2],
            self.target_pos[0], self.target_pos[1], self.target_pos[2],
            dist_to_target,
            tilt,
            float(self.current_disturbance)
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self):
        """Compute reward based on recovery performance"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        # Get orientation
        orientation = state.kinematics_estimated.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation)
        tilt = np.sqrt(roll**2 + pitch**2)
        
        # Get velocity
        vel = state.kinematics_estimated.linear_velocity
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        speed = np.linalg.norm(velocity)
        
        # Distance to target
        dist_to_target = np.linalg.norm(position - self.target_pos)
        
        reward = 0.0
        
        # 1. Survival reward
        reward += 0.1
        
        # 2. Distance reward (closer to target = better)
        if dist_to_target < self.target_tolerance:
            reward += 5.0  # Big bonus for reaching target
        else:
            reward -= dist_to_target * 0.1
        
        # 3. Stability reward (penalize tilt and speed)
        tilt_penalty = -abs(tilt) * 2.0
        speed_penalty = -speed * 0.1 if speed > 5.0 else 0.0
        reward += tilt_penalty + speed_penalty
        
        # 4. Recovery bonus (if recovering from disturbance)
        if self.is_recovering:
            # Reward for reducing tilt
            if tilt < 0.3:  # Less than ~17 degrees
                reward += 3.0
                self.is_recovering = False
                self.episode_stats['recoveries'] += 1
                print(f"✅ Recovery successful at step {self.current_step}!")
        
        # 5. Altitude maintenance
        altitude_error = abs(position[2] - self.target_pos[2])
        if altitude_error < 2.0:
            reward += 1.0
        else:
            reward -= altitude_error * 0.5
        
        # 6. Penalties for bad states
        if position[2] > -1:  # Too close to ground
            reward -= 10.0
        
        if tilt > np.pi/2:  # Upside down
            reward -= 5.0
        
        return float(reward)
    
    def _check_terminated(self):
        """Check if episode should terminate"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        # Crashed (hit ground)
        if pos.z_val > -0.5:
            self.episode_stats['crashes'] += 1
            print(f"💥 CRASHED at step {self.current_step}")
            return True
        
        # Flew too far away
        dist_horizontal = np.sqrt(pos.x_val**2 + pos.y_val**2)
        if dist_horizontal > 100:
            print(f"🚫 OUT OF BOUNDS at step {self.current_step}")
            return True
        
        # Collision detection
        collision = self.client.simGetCollisionInfo()
        if collision.has_collided:
            self.episode_stats['crashes'] += 1
            print(f"💥 COLLISION at step {self.current_step}")
            return True
        
        return False
    
    def _apply_random_disturbance(self):
        """Apply random disturbance to drone"""
        # Curriculum learning: start easy, get harder
        difficulty = min(self.current_step / 500.0, 1.0)
        
        # Choose disturbance type
        disturbance_roll = random.random()
        
        if disturbance_roll < 0.3 * difficulty:
            self._apply_bird_strike(large=False)
            self.current_disturbance = 1
        elif disturbance_roll < 0.5 * difficulty:
            self._apply_bird_strike(large=True)
            self.current_disturbance = 2
        elif disturbance_roll < 0.7:
            self._apply_wind_gust()
            self.current_disturbance = 3
        elif disturbance_roll < 0.85 * difficulty:
            self._apply_extreme_wind()
            self.current_disturbance = 4
        else:
            self._apply_propeller_damage()
            self.current_disturbance = 6
        
        self.disturbance_applied = True
        self.is_recovering = True
        self.disturbance_start_time = self.current_step
        
        print(f"⚡ Disturbance applied: {self.disturbance_types[self.current_disturbance]}")
    
    def _apply_bird_strike(self, large=False):
        """Simulate bird strike"""
        bird_mass = 4.0 if large else 1.0
        
        # Wind gust
        magnitude = bird_mass * 15
        angle = random.uniform(0, 2*np.pi)
        wind = airsim.Vector3r(
            magnitude * np.cos(angle),
            magnitude * np.sin(angle),
            random.uniform(-bird_mass*2, bird_mass)
        )
        self.client.simSetWind(wind)
        
        # Violent tumble
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=random.uniform(-12, 12) if large else random.uniform(-6, 6),
            pitch_rate=random.uniform(-12, 12) if large else random.uniform(-6, 6),
            yaw_rate=random.uniform(-8, 8) if large else random.uniform(-4, 4),
            throttle=0.2 if large else 0.4,
            duration=0.5
        )
        
        # Gradual wind reduction
        time.sleep(0.5)
        self.client.simSetWind(airsim.Vector3r(
            wind.x_val * 0.5, wind.y_val * 0.5, wind.z_val * 0.5
        ))
        time.sleep(0.3)
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
    
    def _apply_wind_gust(self):
        """Apply sudden wind gust"""
        wind = airsim.Vector3r(
            random.uniform(-20, 20),
            random.uniform(-20, 20),
            random.uniform(-5, 5)
        )
        self.client.simSetWind(wind)
        time.sleep(1.0)
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
    
    def _apply_extreme_wind(self):
        """Apply extreme wind conditions"""
        wind = airsim.Vector3r(
            random.uniform(-40, 40),
            random.uniform(-40, 40),
            random.uniform(-10, 10)
        )
        self.client.simSetWind(wind)
        
        # Add tumbling
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=random.uniform(-8, 8),
            pitch_rate=random.uniform(-8, 8),
            yaw_rate=random.uniform(-5, 5),
            throttle=0.3,
            duration=0.8
        )
        
        time.sleep(0.8)
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
    
    def _apply_propeller_damage(self):
        """Simulate propeller damage (asymmetric thrust)"""
        wind = airsim.Vector3r(
            random.uniform(-25, 25),
            random.uniform(-25, 25),
            -10
        )
        self.client.simSetWind(wind)
        
        # Uncontrollable yaw spin
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=random.uniform(-4, 4),
            pitch_rate=random.uniform(-4, 4),
            yaw_rate=random.uniform(-18, 18),  # Extreme yaw
            throttle=0.3,
            duration=1.0
        )
        
        time.sleep(1.0)
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
    
    def _get_info(self):
        """Get additional info for logging"""
        return {
            'episode_step': self.current_step,
            'disturbance_type': self.disturbance_types[self.current_disturbance],
            'is_recovering': self.is_recovering,
            'episode_stats': self.episode_stats.copy()
        }
    
    def render(self, mode='human'):
        """Render environment (not implemented)"""
        pass
    
    def close(self):
        """Clean up environment"""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)