import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import time
import random

class DroneReflexRecoveryEnv(gym.Env):
    """
    HIGH-SPEED Reflex Recovery Environment
    
    Goal: INSTANT recovery from violent flips
    - Ultra-fast control loop (50Hz)
    - Aggressive counter-maneuvers
    - Sub-second recovery time
    - Reward heavily emphasizes SPEED
    """
    
    def __init__(self, ip_address="127.0.0.1"):
        
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # CRITICAL: Ultra-fast control frequency
        self.step_length = 0.02  # 50Hz control! (was 0.1s)
        self.max_episode_steps = 500  # 10 seconds max
        self.current_step = 0
        
        # Training stage
        self.training_stage = 1
        self.total_episodes = 0
        
        # Altitude
        self.nominal_altitude = -20.0
        self.target_pos = np.array([0.0, 0.0, self.nominal_altitude])
        self.ground_level = -1.0
        
        # TIGHT tolerances for fast recovery
        self.altitude_tolerance = 2.0  # ±2m
        self.tilt_tolerance = 0.26  # ~15 degrees
        self.velocity_tolerance = 1.5  # Must be nearly stopped
        
        # Observation: 28 dimensions (added reflex-specific features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(28,),
            dtype=np.float32
        )
        
        # Action space: VERY AGGRESSIVE
        # [roll_rate, pitch_rate, yaw_rate, throttle_delta]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Recovery tracking
        self.disturbance_applied = False
        self.disturbance_step = 0
        self.disturbance_start_time = 0
        self.initial_altitude = self.nominal_altitude
        self.is_recovering = False
        self.recovery_start_step = 0
        self.recovery_start_time = 0
        self.recovery_complete = False
        
        # Performance tracking
        self.max_tilt_this_episode = 0
        self.min_altitude_this_episode = 0
        self.fastest_recovery_time = float('inf')
        
        # Previous action (for action smoothness reward)
        self.prev_action = np.zeros(4)
        
        # Episode stats
        self.episode_stats = self._init_stats()
        
        print(f"✅ Reflex Recovery Environment (50Hz control)")
        print(f"   Step length: {self.step_length}s")
        print(f"   Control frequency: {1/self.step_length:.0f}Hz")
    
    def _init_stats(self):
        return {
            'total_reward': 0,
            'successful_recoveries': 0,
            'crashes': 0,
            'recovery_time': 0,
            'recovery_time_ms': 0,
            'max_tilt': 0,
            'altitude_loss': 0,
            'closest_to_ground': 100,
            'recovery_steps': 0
        }
    
    def set_training_stage(self, stage):
        self.training_stage = stage
        print(f"\n🎓 Stage {stage}: {self._get_stage_description()}")
    
    def _get_stage_description(self):
        descriptions = {
            1: "Hovering only (no disturbances)",
            2: "Light disturbances (learning reflexes)",
            3: "Medium disturbances (building speed)",
            4: "Heavy disturbances (fast recovery)",
            5: "EXTREME (sub-second recovery required)"
        }
        return descriptions.get(self.training_stage, "")
    
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
            velocity=5
        ).join()
        
        time.sleep(0.8)
        
        # Reset tracking
        self.current_step = 0
        self.disturbance_applied = False
        self.is_recovering = False
        self.recovery_complete = False
        self.max_tilt_this_episode = 0
        self.prev_action = np.zeros(4)
        
        state = self.client.getMultirotorState()
        self.initial_altitude = state.kinematics_estimated.position.z_val
        self.min_altitude_this_episode = self.initial_altitude
        
        self.episode_stats = self._init_stats()
        self.total_episodes += 1
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        self.current_step += 1
        
        # ULTRA AGGRESSIVE action scaling based on stage
        action_scale = self._get_action_scale()
        
        # Clip actions for safety but allow extreme values
        action = np.clip(action, -1.0, 1.0)
        
        roll_rate = action[0] * action_scale['roll']
        pitch_rate = action[1] * action_scale['pitch']
        yaw_rate = action[2] * action_scale['yaw']
        
        # Throttle as DELTA from hover point (0.59)
        throttle_delta = action[3] * action_scale['throttle']
        throttle = np.clip(0.59 + throttle_delta, 0.0, 1.0)
        
        # Apply action with MINIMAL delay
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=float(roll_rate),
            pitch_rate=float(pitch_rate),
            yaw_rate=float(yaw_rate),
            throttle=float(throttle),
            duration=self.step_length
        )
        
        time.sleep(self.step_length)
        
        # Apply disturbance EARLY in episode
        if not self.disturbance_applied and self.current_step == 20:  # After 0.4s
            if self._should_apply_disturbance():
                self._apply_disturbance()
        
        # Get state
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_episode_steps
        info = self._get_info()
        
        self.episode_stats['total_reward'] += reward
        
        # Save previous action
        self.prev_action = action.copy()
        
        return obs, reward, terminated, truncated, info
    
    def _get_action_scale(self):
        """VERY AGGRESSIVE action limits for fast recovery"""
        scales = {
            1: {'roll': 5.0,  'pitch': 5.0,  'yaw': 3.0,  'throttle': 0.15},
            2: {'roll': 12.0, 'pitch': 12.0, 'yaw': 6.0,  'throttle': 0.25},
            3: {'roll': 20.0, 'pitch': 20.0, 'yaw': 10.0, 'throttle': 0.35},
            4: {'roll': 30.0, 'pitch': 30.0, 'yaw': 15.0, 'throttle': 0.4},  # EXTREME!
            5: {'roll': 40.0, 'pitch': 40.0, 'yaw': 20.0, 'throttle': 0.45}  # INSANE!
        }
        return scales.get(self.training_stage, scales[3])
    
    def _should_apply_disturbance(self):
        if self.training_stage == 1:
            return False
        elif self.training_stage == 2:
            return random.random() < 0.5
        else:
            return random.random() < 0.8
    
    def _apply_disturbance(self):
        """Apply INSTANT violent disturbance"""
        print(f"\n⚡ DISTURBANCE at t={self.current_step*self.step_length:.2f}s (Stage {self.training_stage})")
        
        self.disturbance_applied = True
        self.disturbance_step = self.current_step
        self.disturbance_start_time = time.time()
        self.is_recovering = True
        self.recovery_start_step = self.current_step
        self.recovery_start_time = time.time()
        
        state = self.client.getMultirotorState()
        self.initial_altitude = state.kinematics_estimated.position.z_val
        
        if self.training_stage == 2:
            self._light_disturbance()
        elif self.training_stage == 3:
            self._medium_disturbance()
        elif self.training_stage == 4:
            self._heavy_disturbance()
        else:
            self._extreme_disturbance()
    
    def _light_disturbance(self):
        """Quick jolt"""
        wind = airsim.Vector3r(
            random.uniform(-15, 15),
            random.uniform(-15, 15),
            random.uniform(-5, 5)
        )
        self.client.simSetWind(wind)
        
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=random.uniform(-5, 5),
            pitch_rate=random.uniform(-5, 5),
            yaw_rate=random.uniform(-3, 3),
            throttle=0.5,
            duration=0.3
        )
        
        time.sleep(0.3)
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
    
    def _medium_disturbance(self):
        """Fast flip"""
        magnitude = random.uniform(30, 45)
        angle = random.uniform(0, 2*np.pi)
        
        wind = airsim.Vector3r(
            magnitude * np.cos(angle),
            magnitude * np.sin(angle),
            random.uniform(-10, 8)
        )
        self.client.simSetWind(wind)
        
        # Quick violent spin
        self.client.moveByAngleRatesThrottleAsync(
            roll_rate=random.uniform(-15, 15),
            pitch_rate=random.uniform(-15, 15),
            yaw_rate=random.uniform(-10, 10),
            throttle=0.35,
            duration=0.4
        )
        
        time.sleep(0.4)
        
        # Quick wind reduction
        self.client.simSetWind(airsim.Vector3r(
            wind.x_val * 0.2, wind.y_val * 0.2, wind.z_val * 0.2
        ))
        time.sleep(0.1)
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
    
    def _heavy_disturbance(self):
        """Violent tumble - requires FAST recovery"""
        magnitude = random.uniform(50, 70)
        angle = random.uniform(0, 2*np.pi)
        
        wind = airsim.Vector3r(
            magnitude * np.cos(angle),
            magnitude * np.sin(angle),
            random.uniform(-15, 10)
        )
        self.client.simSetWind(wind)
        
        # Multiple rapid tumbles
        for _ in range(2):
            self.client.moveByAngleRatesThrottleAsync(
                roll_rate=random.uniform(-20, 20),
                pitch_rate=random.uniform(-20, 20),
                yaw_rate=random.uniform(-12, 12),
                throttle=random.uniform(0.2, 0.4),
                duration=0.25
            )
            time.sleep(0.25)
        
        # Fast wind decay
        for i in range(2):
            factor = 0.5 - (i * 0.4)
            self.client.simSetWind(airsim.Vector3r(
                wind.x_val * factor, wind.y_val * factor, wind.z_val * factor
            ))
            time.sleep(0.1)
        
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
    
    def _extreme_disturbance(self):
        """INSANE tumble - expert reflexes required"""
        magnitude = random.uniform(70, 90)
        angle = random.uniform(0, 2*np.pi)
        
        wind = airsim.Vector3r(
            magnitude * np.cos(angle),
            magnitude * np.sin(angle),
            random.uniform(-20, 15)
        )
        self.client.simSetWind(wind)
        
        # Chaotic multi-axis tumbling
        for _ in range(3):
            self.client.moveByAngleRatesThrottleAsync(
                roll_rate=random.uniform(-25, 25),
                pitch_rate=random.uniform(-25, 25),
                yaw_rate=random.uniform(-15, 15),
                throttle=random.uniform(0.1, 0.35),
                duration=0.2
            )
            time.sleep(0.2)
        
        # Rapid wind removal
        for i in range(3):
            factor = 0.6 - (i * 0.3)
            self.client.simSetWind(airsim.Vector3r(
                wind.x_val * factor, wind.y_val * factor, wind.z_val * factor
            ))
            time.sleep(0.05)
        
        self.client.simSetWind(airsim.Vector3r(0, 0, 0))
    
    def _get_obs(self):
        """Observation with reflex-critical features"""
        state = self.client.getMultirotorState()
        
        pos = state.kinematics_estimated.position
        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        orientation = state.kinematics_estimated.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation)
        
        vel = state.kinematics_estimated.linear_velocity
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        
        ang_vel = state.kinematics_estimated.angular_velocity
        angular_velocity = np.array([ang_vel.x_val, ang_vel.y_val, ang_vel.z_val])
        
        # Reflex features
        altitude_error = position[2] - self.nominal_altitude
        tilt = np.sqrt(roll**2 + pitch**2)
        speed = np.linalg.norm(velocity)
        angular_speed = np.linalg.norm([ang_vel.x_val, ang_vel.y_val, ang_vel.z_val])
        
        ground_distance = abs(position[2] - self.ground_level)
        
        # Time-critical features
        time_since_disturbance = (self.current_step - self.disturbance_step) * self.step_length if self.disturbance_applied else 0
        
        # Falling metrics
        falling_speed = velocity[2]  # Positive = falling
        vertical_acceleration = falling_speed  # Approximation
        
        # Recovery urgency (exponential with ground proximity)
        if ground_distance < 10:
            recovery_urgency = 1.0 - (ground_distance / 10.0) ** 2
        else:
            recovery_urgency = 0.0
        
        # Time to impact (if falling)
        if falling_speed > 0.5:
            time_to_impact = ground_distance / falling_speed
        else:
            time_to_impact = 100.0
        
        # Track extremes
        if tilt > self.max_tilt_this_episode:
            self.max_tilt_this_episode = tilt
        if position[2] > self.min_altitude_this_episode:
            self.min_altitude_this_episode = position[2]
        
        # Stability indicator
        is_stable = float(self._is_stable())
        
        # Rate of tilt change (important for predicting recovery)
        tilt_rate = angular_speed
        
        obs = np.array([
            # Position & orientation
            position[0], position[1], position[2],
            roll, pitch, yaw,
            # Velocities
            velocity[0], velocity[1], velocity[2],
            angular_velocity[0], angular_velocity[1], angular_velocity[2],
            # Reflex-critical features
            altitude_error, tilt, speed, angular_speed,
            ground_distance, falling_speed, recovery_urgency, time_to_impact,
            # Recovery tracking
            float(self.is_recovering), time_since_disturbance, tilt_rate,
            # Context
            float(self.training_stage), is_stable,
            # Previous action (for temporal awareness)
            self.prev_action[0], self.prev_action[1]
        ], dtype=np.float32)
        
        return obs
    
    def _is_stable(self):
        """Check if stable"""
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
    
    def _compute_reward(self, action):
        """Reward function HEAVILY emphasizing SPEED"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        orientation = state.kinematics_estimated.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation)
        tilt = np.sqrt(roll**2 + pitch**2)
        
        vel = state.kinematics_estimated.linear_velocity
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        speed = np.linalg.norm(velocity)
        falling_speed = velocity[2]
        
        ang_vel = state.kinematics_estimated.angular_velocity
        angular_speed = np.linalg.norm([ang_vel.x_val, ang_vel.y_val, ang_vel.z_val])
        
        reward = 0.0
        
        # === SURVIVAL (top priority) ===
        ground_distance = abs(position[2] - self.ground_level)
        
        if ground_distance < 2.0:
            reward -= 100.0 * (1.0 - ground_distance / 2.0) ** 2  # Exponential penalty
        elif ground_distance < 5.0:
            reward -= 20.0 * (1.0 - ground_distance / 5.0)
        
        # === RECOVERY SPEED (critical!) ===
        if self.is_recovering:
            recovery_steps = self.current_step - self.recovery_start_step
            
            # Penalty for EVERY step not recovered
            reward -= 1.0  # Strong time pressure!
            
            # Extra penalty if taking too long
            if recovery_steps > 25:  # More than 0.5s
                reward -= 2.0
            if recovery_steps > 50:  # More than 1.0s
                reward -= 5.0
            
            # Check if just recovered
            if self._is_stable() and not self.recovery_complete:
                recovery_time_ms = recovery_steps * self.step_length * 1000
                
                # MASSIVE reward for fast recovery
                if recovery_steps < 15:  # Under 0.3s
                    reward += 100.0
                    print(f"🚀 LIGHTNING FAST! {recovery_time_ms:.0f}ms")
                elif recovery_steps < 25:  # Under 0.5s
                    reward += 70.0
                    print(f"⚡ VERY FAST! {recovery_time_ms:.0f}ms")
                elif recovery_steps < 40:  # Under 0.8s
                    reward += 40.0
                    print(f"✅ FAST! {recovery_time_ms:.0f}ms")
                elif recovery_steps < 60:  # Under 1.2s
                    reward += 20.0
                    print(f"✓ Recovered in {recovery_time_ms:.0f}ms")
                else:
                    reward += 5.0
                    print(f"△ Slow recovery: {recovery_time_ms:.0f}ms")
                
                self.is_recovering = False
                self.recovery_complete = True
                self.episode_stats['successful_recoveries'] += 1
                self.episode_stats['recovery_time'] = recovery_steps * self.step_length
                self.episode_stats['recovery_time_ms'] = recovery_time_ms
                self.episode_stats['recovery_steps'] = recovery_steps
        
        # === TILT CORRECTION (during recovery) ===
        if self.disturbance_applied:
            # Reward for reducing tilt AGGRESSIVELY
            if tilt < 0.2:  # Nearly level
                reward += 5.0
            elif tilt < 0.4:
                reward += 2.0
            elif tilt < 0.7:
                reward += 0.5
            else:
                reward -= tilt * 3.0  # Penalty for being tilted
        
        # === ALTITUDE MAINTENANCE ===
        altitude_error = abs(position[2] - self.nominal_altitude)
        
        if altitude_error < 1.0:
            reward += 3.0
        elif altitude_error < 2.5:
            reward += 1.0
        else:
            reward -= altitude_error * 0.5
        
        # === VELOCITY CONTROL ===
        if speed < 1.0:
            reward += 2.0
        elif speed < 3.0:
            reward += 0.5
        else:
            reward -= speed * 0.2
        
        # === FALLING PENALTY ===
        if falling_speed > 5.0:
            reward -= falling_speed * 1.5
        elif falling_speed > 2.0:
            reward -= falling_speed * 0.5
        
        # === ANGULAR VELOCITY (want it low after recovery) ===
        if not self.is_recovering and angular_speed > 2.0:
            reward -= angular_speed * 0.3
        
        # === ACTION SMOOTHNESS (penalize jerky movements AFTER recovery) ===
        if self.recovery_complete:
            action_diff = np.linalg.norm(action - self.prev_action)
            if action_diff > 0.5:
                reward -= 0.5
        
        # === BASE SURVIVAL ===
        reward += 0.1
        
        return float(reward)
    
    def _check_terminated(self):
        """Termination check"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        # Crashed
        if pos.z_val > self.ground_level:
            self.episode_stats['crashes'] += 1
            altitude_lost = self.initial_altitude - pos.z_val
            print(f"💥 CRASHED! Lost {altitude_lost:.1f}m")
            return True
        
        # Flew away
        if np.sqrt(pos.x_val**2 + pos.y_val**2) > 35:
            return True
        
        # Collision
        if self.client.simGetCollisionInfo().has_collided:
            self.episode_stats['crashes'] += 1
            return True
        
        return False
    
    def _get_info(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        return {
            'episode_step': self.current_step,
            'time_elapsed': self.current_step * self.step_length,
            'training_stage': self.training_stage,
            'is_recovering': self.is_recovering,
            'recovery_complete': self.recovery_complete,
            'current_altitude': -pos.z_val,
            'ground_distance': abs(pos.z_val - self.ground_level),
            'max_tilt_this_episode': self.max_tilt_this_episode,
            'episode_stats': self.episode_stats.copy()
        }
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)