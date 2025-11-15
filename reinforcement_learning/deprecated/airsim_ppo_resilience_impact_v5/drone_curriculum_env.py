import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import time
import random

class DroneCurriculumRecoveryEnv(gym.Env):
    """
    Curriculum Learning Environment for Drone Recovery
    
    Stage 1 (0-100k): Learn basic stability and hovering (NO disturbances)
    Stage 2 (100k-300k): Small disturbances (light wind, small tilts)
    Stage 3 (300k-600k): Medium disturbances (wind gusts, moderate flips)
    Stage 4 (600k+): Full disturbances (bird strikes, violent flips)
    """
    
    def __init__(self, ip_address="127.0.0.1"):
        
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Training stage (will be updated externally)
        self.training_stage = 1
        self.total_episodes = 0
        
        # Control frequency
        self.step_length = 0.1  # 10Hz
        self.max_episode_steps = 300
        self.current_step = 0
        
        # Altitude settings
        self.nominal_altitude = -20.0  # Start at 20m (lower than before)
        self.target_pos = np.array([0.0, 0.0, self.nominal_altitude])
        self.ground_level = -1.0
        
        # Tolerances
        self.altitude_tolerance = 3.0
        self.tilt_tolerance = 0.35  # ~20 degrees
        self.velocity_tolerance = 2.5
        
        # Observation space: 24 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32
        )
        
        # Action space: [roll_rate, pitch_rate, yaw_rate, throttle]
        # Start conservative, increase with curriculum
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.disturbance_applied = False
        self.disturbance_step = 0
        self.initial_altitude = self.nominal_altitude
        self.is_recovering = False
        self.recovery_start_step = 0
        self.consecutive_stable_steps = 0
        
        # Statistics
        self.episode_stats = self._init_stats()
        
        print(f"✅ Curriculum Environment initialized (Stage {self.training_stage})")
    
    def _init_stats(self):
        return {
            'total_reward': 0,
            'successful_recoveries': 0,
            'crashes': 0,
            'recovery_time': 0,
            'max_tilt': 0,
            'altitude_loss': 0,
            'closest_to_ground': 100,
            'stable_time': 0
        }
    
    def set_training_stage(self, stage):
        """Update training curriculum stage"""
        self.training_stage = stage
        print(f"\n🎓 Training Stage Updated: {stage}")
        print(self._get_stage_description())
    
    def _get_stage_description(self):
        """Get description of current training stage"""
        descriptions = {
            1: "Stage 1: Basic hovering and stability (NO disturbances)",
            2: "Stage 2: Light disturbances (gentle winds, small tilts)",
            3: "Stage 3: Medium disturbances (wind gusts, moderate flips)",
            4: "Stage 4: Heavy disturbances (bird strikes, violent flips)",
            5: "Stage 5: EXTREME (catastrophic failures)"
        }
        return descriptions.get(self.training_stage, "Unknown stage")
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset drone
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Takeoff
        self.client.takeoffAsync().join()
        self.client.moveToPositionAsync(
            self.target_pos[0],
            self.target_pos[1], 
            self.target_pos[2],
            velocity=3
        ).join()
        
        time.sleep(1.0)
        
        # Reset variables
        self.current_step = 0
        self.disturbance_applied = False
        self.is_recovering = False
        self.consecutive_stable_steps = 0
        
        state = self.client.getMultirotorState()
        self.initial_altitude = state.kinematics_estimated.position.z_val
        
        self.episode_stats = self._init_stats()
        self.total_episodes += 1
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        self.current_step += 1
        
        # Scale actions based on training stage
        action_scale = self._get_action_scale()
        
        roll_rate = action[0] * action_scale['roll']
        pitch_rate = action[1] * action_scale['pitch']
        yaw_rate = action[2] * action_scale['yaw']
        throttle = np.clip((action[3] + 1) * 0.5, 0.0, 1.0)
        
        # Apply action
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=float(roll_rate),
            pitch_rate=float(pitch_rate),
            yaw_rate=float(yaw_rate),
            throttle=float(throttle),
            duration=self.step_length
        )
        
        time.sleep(self.step_length)
        
        # Apply curriculum-based disturbance
        if not self.disturbance_applied and self.current_step == 15:
            if self._should_apply_disturbance():
                self._apply_curriculum_disturbance()
        
        # Get observation and compute reward
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_episode_steps
        info = self._get_info()
        
        self.episode_stats['total_reward'] += reward
        
        # Check if stable
        if self._is_stable():
            self.consecutive_stable_steps += 1
        else:
            self.consecutive_stable_steps = 0
        
        return obs, reward, terminated, truncated, info
    
    def _get_action_scale(self):
        """Scale action limits based on training stage"""
        scales = {
            1: {'roll': 3.0,  'pitch': 3.0,  'yaw': 2.0},   # Very gentle
            2: {'roll': 6.0,  'pitch': 6.0,  'yaw': 3.0},   # Gentle
            3: {'roll': 10.0, 'pitch': 10.0, 'yaw': 5.0},   # Moderate
            4: {'roll': 15.0, 'pitch': 15.0, 'yaw': 8.0},   # Aggressive
            5: {'roll': 20.0, 'pitch': 20.0, 'yaw': 12.0}   # Very aggressive
        }
        return scales.get(self.training_stage, scales[3])
    
    def _should_apply_disturbance(self):
        """Decide if disturbance should be applied based on stage"""
        if self.training_stage == 1:
            return False  # NO disturbances in stage 1
        elif self.training_stage == 2:
            return random.random() < 0.3  # 30% chance
        elif self.training_stage == 3:
            return random.random() < 0.5  # 50% chance
        else:
            return random.random() < 0.7  # 70% chance
    
    def _apply_curriculum_disturbance(self):
        """Apply disturbance appropriate for current stage"""
        print(f"\n⚡ Disturbance at step {self.current_step} (Stage {self.training_stage})")
        
        self.disturbance_applied = True
        self.disturbance_step = self.current_step
        self.is_recovering = True
        self.recovery_start_step = self.current_step + 1
        
        state = self.client.getMultirotorState()
        self.initial_altitude = state.kinematics_estimated.position.z_val
        
        if self.training_stage == 2:
            self._apply_stage2_disturbance()
        elif self.training_stage == 3:
            self._apply_stage3_disturbance()
        elif self.training_stage == 4:
            self._apply_stage4_disturbance()
        else:
            self._apply_stage5_disturbance()
    
    def _apply_stage2_disturbance(self):
        """Light disturbance - learning basics"""
        print("   Type: Light wind gust + small tilt")
        
        # Gentle wind
        wind = airsim.Vector3r(
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-3, 3)
        )
        self.client.simSetWind(wind)
        
        # Small tilt
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=random.uniform(-3, 3),
            pitch_rate=random.uniform(-3, 3),
            yaw_rate=random.uniform(-2, 2),
            throttle=0.5,
            duration=0.5
        )
        
        time.sleep(0.5)
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
    
    def _apply_stage3_disturbance(self):
        """Medium disturbance - building skills"""
        print("   Type: Wind gust + moderate flip")
        
        # Moderate wind
        wind = airsim.Vector3r(
            random.uniform(-25, 25),
            random.uniform(-25, 25),
            random.uniform(-8, 8)
        )
        self.client.simSetWind(wind)
        
        # Moderate tumble
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=random.uniform(-8, 8),
            pitch_rate=random.uniform(-8, 8),
            yaw_rate=random.uniform(-5, 5),
            throttle=0.4,
            duration=0.6
        )
        
        time.sleep(0.6)
        
        # Decay wind
        self.client.simSetWind(airsim.Vector3r(
            wind.x_val * 0.3, wind.y_val * 0.3, wind.z_val * 0.3
        ))
        time.sleep(0.3)
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
    
    def _apply_stage4_disturbance(self):
        """Heavy disturbance - real challenge"""
        disturbance_type = random.choice(['bird_strike', 'severe_wind', 'prop_damage'])
        print(f"   Type: {disturbance_type}")
        
        if disturbance_type == 'bird_strike':
            # Large bird strike
            magnitude = random.uniform(40, 60)
            angle = random.uniform(0, 2*np.pi)
            
            wind = airsim.Vector3r(
                magnitude * np.cos(angle),
                magnitude * np.sin(angle),
                random.uniform(-15, 10)
            )
            self.client.simSetWind(wind)
            
            # Violent tumble
            self.client.moveByAngleRatesThrottleAsync(
                roll_rate=random.uniform(-12, 12),
                pitch_rate=random.uniform(-12, 12),
                yaw_rate=random.uniform(-8, 8),
                throttle=0.2,
                duration=0.7
            )
            
            time.sleep(0.7)
            
            # Gradual wind reduction
            for i in range(3):
                factor = 1.0 - ((i+1) * 0.3)
                self.client.simSetWind(airsim.Vector3r(
                    wind.x_val * factor, wind.y_val * factor, wind.z_val * factor
                ))
                time.sleep(0.2)
            
            self.client.simSetWind(airsim.Vector3r(0, 0, 0))
            
        elif disturbance_type == 'severe_wind':
            # Multiple wind gusts
            for _ in range(2):
                wind = airsim.Vector3r(
                    random.uniform(-40, 40),
                    random.uniform(-40, 40),
                    random.uniform(-10, 10)
                )
                self.client.simSetWind(wind)
                
                self.client.moveByAngleRatesThrottleAsync(
                    roll_rate=random.uniform(-10, 10),
                    pitch_rate=random.uniform(-10, 10),
                    yaw_rate=random.uniform(-6, 6),
                    throttle=0.3,
                    duration=0.4
                )
                time.sleep(0.4)
            
            self.client.simSetWind(airsim.Vector3r(0, 0, 0))
            
        else:  # prop_damage
            # Asymmetric thrust
            wind = airsim.Vector3r(
                random.uniform(-35, 35),
                random.uniform(-35, 35),
                -10
            )
            self.client.simSetWind(wind)
            
            # Extreme yaw
            for _ in range(3):
                self.client.moveByAngleRatesThrottleAsync(
                    roll_rate=random.uniform(-6, 6),
                    pitch_rate=random.uniform(-6, 6),
                    yaw_rate=random.uniform(-18, 18),
                    throttle=0.25,
                    duration=0.4
                )
                time.sleep(0.4)
            
            self.client.simSetWind(airsim.Vector3r(0, 0, 0))
    
    def _apply_stage5_disturbance(self):
        """EXTREME disturbance - expert level"""
        print("   Type: CATASTROPHIC failure")
        
        # Massive wind
        magnitude = random.uniform(60, 80)
        angle = random.uniform(0, 2*np.pi)
        
        wind = airsim.Vector3r(
            magnitude * np.cos(angle),
            magnitude * np.sin(angle),
            random.uniform(-20, 15)
        )
        self.client.simSetWind(wind)
        
        # Multiple chaotic tumbles
        for _ in range(3):
            self.client.moveByAngleRatesThrottleAsync(
                roll_rate=random.uniform(-18, 18),
                pitch_rate=random.uniform(-18, 18),
                yaw_rate=random.uniform(-12, 12),
                throttle=random.uniform(0.1, 0.3),
                duration=0.4
            )
            time.sleep(0.4)
        
        # Gradual wind decay
        for i in range(4):
            factor = 1.0 - ((i+1) * 0.25)
            self.client.simSetWind(airsim.Vector3r(
                wind.x_val * factor, wind.y_val * factor, wind.z_val * factor
            ))
            time.sleep(0.2)
        
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
    
    def _get_obs(self):
        """Get observation"""
        state = self.client.getMultirotorState()
        
        pos = state.kinematics_estimated.position
        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        orientation = state.kinematics_estimated.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation)
        
        vel = state.kinematics_estimated.linear_velocity
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        
        ang_vel = state.kinematics_estimated.angular_velocity
        angular_velocity = np.array([ang_vel.x_val, ang_vel.y_val, ang_vel.z_val])
        
        # Derived features
        altitude_error = position[2] - self.nominal_altitude
        tilt = np.sqrt(roll**2 + pitch**2)
        speed = np.linalg.norm(velocity)
        ground_distance = abs(position[2] - self.ground_level)
        
        is_stable = float(self._is_stable())
        time_since_disturbance = (self.current_step - self.disturbance_step) * 0.1 if self.disturbance_applied else 0
        falling_speed = velocity[2]
        recovery_urgency = max(0, 1.0 - ground_distance / 15.0)
        
        # Track stats
        if tilt > self.episode_stats['max_tilt']:
            self.episode_stats['max_tilt'] = tilt
        if ground_distance < self.episode_stats['closest_to_ground']:
            self.episode_stats['closest_to_ground'] = ground_distance
        
        obs = np.array([
            position[0], position[1], position[2],
            roll, pitch, yaw,
            velocity[0], velocity[1], velocity[2],
            angular_velocity[0], angular_velocity[1], angular_velocity[2],
            altitude_error, tilt, speed,
            ground_distance, is_stable, time_since_disturbance,
            falling_speed, recovery_urgency,
            float(self.training_stage),
            float(self.disturbance_applied),
            float(self.is_recovering),
            float(self.consecutive_stable_steps)
        ], dtype=np.float32)
        
        return obs
    
    def _is_stable(self):
        """Check if drone is stable"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        altitude_error = abs(pos.z_val - self.nominal_altitude)
        
        orientation = state.kinematics_estimated.orientation
        roll, pitch, _ = airsim.to_eularian_angles(orientation)
        tilt = np.sqrt(roll**2 + pitch**2)
        
        vel = state.kinematics_estimated.linear_velocity
        speed = np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])
        
        return (altitude_error < self.altitude_tolerance and
                tilt < self.tilt_tolerance and
                speed < self.velocity_tolerance)
    
    def _compute_reward(self):
        """Curriculum-based reward function"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        orientation = state.kinematics_estimated.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation)
        tilt = np.sqrt(roll**2 + pitch**2)
        
        vel = state.kinematics_estimated.linear_velocity
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        speed = np.linalg.norm(velocity)
        
        reward = 0.0
        
        # Base survival reward
        reward += 0.5
        
        # Ground proximity (CRITICAL)
        ground_distance = abs(position[2] - self.ground_level)
        if ground_distance < 3.0:
            reward -= 30.0 * (1.0 - ground_distance / 3.0)
        
        # Altitude maintenance
        altitude_error = abs(position[2] - self.nominal_altitude)
        if altitude_error < 1.5:
            reward += 3.0
        elif altitude_error < 3.0:
            reward += 1.0
        else:
            reward -= altitude_error * 1.0
        
        # Stability rewards (important for stage 1)
        if tilt < 0.17:  # < 10 degrees
            reward += 2.0
        elif tilt < 0.35:  # < 20 degrees
            reward += 1.0
        elif tilt < 0.7:  # < 40 degrees
            reward -= 1.0
        else:
            reward -= 5.0
        
        # Speed control
        if speed < 1.0:
            reward += 1.5
        elif speed < 3.0:
            reward += 0.5
        else:
            reward -= speed * 0.3
        
        # Consecutive stability bonus (helps with hovering)
        if self.consecutive_stable_steps > 20:
            reward += 5.0
        elif self.consecutive_stable_steps > 10:
            reward += 2.0
        
        # Recovery bonus
        if self.is_recovering and self._is_stable():
            recovery_time = self.current_step - self.recovery_start_step
            time_bonus = max(30.0 - recovery_time * 0.5, 5.0)
            reward += time_bonus
            
            self.is_recovering = False
            self.episode_stats['successful_recoveries'] += 1
            self.episode_stats['recovery_time'] = recovery_time
            
            print(f"✅ RECOVERY at step {self.current_step} ({recovery_time} steps)")
        
        # Time penalty during recovery
        if self.is_recovering:
            reward -= 0.3
        
        return float(reward)
    
    def _check_terminated(self):
        """Check termination"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        # Crashed
        if pos.z_val > self.ground_level:
            self.episode_stats['crashes'] += 1
            altitude_lost = self.initial_altitude - pos.z_val
            print(f"💥 CRASHED! Lost {altitude_lost:.1f}m altitude")
            return True
        
        # Flew away
        horizontal_dist = np.sqrt(pos.x_val**2 + pos.y_val**2)
        if horizontal_dist > 40:
            return True
        
        # Collision
        collision = self.client.simGetCollisionInfo()
        if collision.has_collided:
            self.episode_stats['crashes'] += 1
            return True
        
        return False
    
    def _get_info(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        return {
            'episode_step': self.current_step,
            'training_stage': self.training_stage,
            'disturbance_applied': self.disturbance_applied,
            'is_recovering': self.is_recovering,
            'current_altitude': -pos.z_val,
            'ground_distance': abs(pos.z_val - self.ground_level),
            'is_stable': self._is_stable(),
            'consecutive_stable': self.consecutive_stable_steps,
            'episode_stats': self.episode_stats.copy()
        }
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)