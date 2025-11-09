"""
DRONE FLIP RECOVERY ENVIRONMENT - MOTOR-LEVEL TUMBLE (13 OBSERVATIONS)
=======================================================================
Stage 3: Simulates bird attacks using MOTOR-LEVEL disturbances

KEY DIFFERENCE: Instead of trying to set orientation (which AirSim ignores),
we apply ASYMMETRIC MOTOR FORCES to create real physical rotation!

The drone must learn to:
1. Detect tumbling (using angular velocity - spinning fast!)
2. Apply counter-rotation through motor commands
3. Orient itself upright
4. Stabilize and return to hover position
5. Handle wind during recovery

Observation Space: (13,)
    - position: [x, y, z]
    - velocity: [vx, vy, vz]  
    - orientation: [qw, qx, qy, qz]
    - angular_velocity: [wx, wy, wz] ← CRITICAL!

Action Space: (3,)
    - velocity commands: [vx, vy, vz] in range [-5, 5] m/s
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time


class DroneFlipRecoveryEnv(gym.Env):
    """
    Drone flip recovery with REAL physics-based tumbling
    """
    
    def __init__(self, target_altitude=10.0, max_steps=500, 
                 wind_strength=5.0, flip_prob=0.5, debug=False):
        super().__init__()
        
        self.target_altitude = target_altitude
        self.max_steps = max_steps
        self.wind_strength = wind_strength
        self.tumble_prob = flip_prob
        self.debug = debug
        
        # Action space: velocity commands
        self.action_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Observation space: 13 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # Episode tracking
        self.episode_steps = 0
        self.stable_steps = 0
        self.current_wind = np.zeros(3)
        self.wind_change_timer = 0
        self.wind_change_interval = np.random.uniform(1.0, 3.0)
        
        # Tumble tracking
        self.tumble_initiated = False
        self.tumble_recovery_started = False
        self.tumble_recovered = False
        self.tumble_start_step = 0
        self.recovery_steps = 0
        self.tumble_duration = 0
        self.tumble_force = np.zeros(3)  # Tumble forces instead of angular velocity
        
        if self.debug:
            print(f"✓ Drone Flip Recovery Environment (MOTOR-LEVEL CONTROL)")
            print(f"  - Target: [0, 0, {-self.target_altitude}]")
            print(f"  - Max steps: {self.max_steps}")
            print(f"  - Wind strength: 0-{self.wind_strength} m/s")
            print(f"  - Tumble probability: {self.tumble_prob*100:.0f}%")
            print(f"  - Mode: MOTOR-LEVEL TUMBLE")
    
    def _get_wind(self):
        """Generate random wind vector"""
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        strength = np.random.uniform(0, self.wind_strength)
        
        wind_x = strength * np.sin(phi) * np.cos(theta)
        wind_y = strength * np.sin(phi) * np.sin(theta)
        wind_z = strength * np.cos(phi) * 0.3
        
        return np.array([wind_x, wind_y, wind_z])
    
    def _apply_wind(self):
        """Apply wind to the drone"""
        self.client.simSetWind(airsim.Vector3r(
            float(self.current_wind[0]),
            float(self.current_wind[1]),
            float(self.current_wind[2])
        ))
    
    def _initiate_tumble(self):
        """
        Simulate bird attack using motor-level disturbance
        """
        # High tumble forces (will be applied as roll/pitch/yaw rates)
        tumble_intensity = np.random.uniform(3.0, 5.0)  # Dramatic but controllable
        
        # Random tumble direction
        self.tumble_force = np.array([
            np.random.uniform(-tumble_intensity, tumble_intensity),  # Roll
            np.random.uniform(-tumble_intensity, tumble_intensity),  # Pitch
            np.random.uniform(-tumble_intensity * 0.5, tumble_intensity * 0.5)  # Yaw
        ])
        
        # Tumble duration (50 steps = 2.5 seconds)
        self.tumble_duration = 50
        
        self.tumble_initiated = True
        self.tumble_start_step = self.episode_steps
        
        if self.debug:
            print(f"   🐦 BIRD ATTACK! Motor disturbance applied!")
            print(f"      Tumble forces: Roll={self.tumble_force[0]:.2f}, "
                  f"Pitch={self.tumble_force[1]:.2f}, Yaw={self.tumble_force[2]:.2f}")
            print(f"      ⚠️  Drone will tumble for ~{self.tumble_duration * 0.05:.1f}s!")
    
    def _is_upright(self, orientation):
        """Check if drone is upright"""
        qw, qx, qy, qz = orientation
        up_z = 1 - 2 * (qx * qx + qy * qy)
        return up_z > 0.7
    
    def _is_tumbling(self, angular_velocity):
        """Check if drone is tumbling"""
        ang_vel_magnitude = np.linalg.norm(angular_velocity)
        return ang_vel_magnitude > 1.0
    
    def _get_observation(self):
        """Get current state (13 observations)"""
        drone_state = self.client.getMultirotorState()
        
        pos = drone_state.kinematics_estimated.position
        vel = drone_state.kinematics_estimated.linear_velocity
        ori = drone_state.kinematics_estimated.orientation
        ang_vel = drone_state.kinematics_estimated.angular_velocity
        
        obs = np.array([
            pos.x_val, pos.y_val, pos.z_val,
            vel.x_val, vel.y_val, vel.z_val,
            ori.w_val, ori.x_val, ori.y_val, ori.z_val,
            ang_vel.x_val, ang_vel.y_val, ang_vel.z_val
        ], dtype=np.float32)
        
        return obs
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Reset AirSim
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Takeoff
        self.client.takeoffAsync().join()
        time.sleep(0.5)
        
        # Move to starting position
        start_x = np.random.uniform(-1, 1)
        start_y = np.random.uniform(-1, 1)
        self.client.moveToPositionAsync(
            start_x, start_y, -self.target_altitude, 5
        ).join()
        time.sleep(1.0)
        
        # Initialize wind
        self.current_wind = self._get_wind()
        self._apply_wind()
        
        # Reset tracking
        self.episode_steps = 0
        self.stable_steps = 0
        self.wind_change_timer = 0
        self.wind_change_interval = np.random.uniform(1.0, 3.0)
        
        # Tumble tracking
        self.tumble_initiated = False
        self.tumble_recovery_started = False
        self.tumble_recovered = False
        self.tumble_start_step = 0
        self.recovery_steps = 0
        self.tumble_force = np.zeros(3)
        
        # Decide if tumble will happen
        self.will_tumble = np.random.random() < self.tumble_prob
        if self.will_tumble:
            self.tumble_trigger_step = np.random.randint(20, 50)
            if self.debug:
                print(f"   ⚠️  Tumble scheduled for step {self.tumble_trigger_step}")
        else:
            self.tumble_trigger_step = -1
            if self.debug:
                print(f"   ✅ No tumble this episode")
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step"""
        self.episode_steps += 1
        
        # Trigger tumble if scheduled
        if self.will_tumble and not self.tumble_initiated:
            if self.episode_steps >= self.tumble_trigger_step:
                self._initiate_tumble()
        
        # Apply tumble disturbance using motor-level commands
        if self.tumble_initiated and not self.tumble_recovered:
            steps_since_tumble = self.episode_steps - self.tumble_start_step
            
            # Apply decaying tumble force
            decay_factor = max(0, 1.0 - (steps_since_tumble / self.tumble_duration))
            current_tumble = self.tumble_force * decay_factor
            
            # Apply tumble as roll/pitch/yaw rates
            try:
                self.client.moveByRollPitchYawThrottleAsync(
                    roll=float(np.clip(current_tumble[0] * 0.2, -1, 1)),
                    pitch=float(np.clip(current_tumble[1] * 0.2, -1, 1)),
                    yaw_rate=float(np.clip(current_tumble[2], -5, 5)),
                    throttle=0.59,
                    duration=0.05
                ).join()
            except:
                pass
            
            # Check for recovery
            obs = self._get_observation()
            ang_vel = obs[10:13]
            ori = obs[6:10]
            
            if np.linalg.norm(ang_vel) < 1.0 and self._is_upright(ori):
                if not self.tumble_recovered:
                    self.tumble_recovered = True
                    self.recovery_steps = self.episode_steps - self.tumble_start_step
                    if self.debug:
                        print(f"   ✅ RECOVERED! Took {self.recovery_steps} steps ({self.recovery_steps * 0.05:.1f}s)")
        else:
            # Normal velocity control
            action = np.clip(action, -5.0, 5.0)
            self.client.moveByVelocityAsync(
                float(action[0]),
                float(action[1]),
                float(action[2]),
                duration=0.05,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0)
            ).join()
        
        # Update wind periodically
        self.wind_change_timer += 1
        if self.wind_change_timer >= int(self.wind_change_interval / 0.05):
            self.current_wind = self._get_wind()
            self._apply_wind()
            self.wind_change_timer = 0
            self.wind_change_interval = np.random.uniform(1.0, 3.0)
        
        # Get observation
        obs = self._get_observation()
        
        # Parse observation
        pos = obs[0:3]
        vel = obs[3:6]
        ori = obs[6:10]
        ang_vel = obs[10:13]
        
        # Calculate metrics
        alt = -pos[2]
        dist_from_center = np.linalg.norm(pos[0:2])
        dist_from_target_alt = abs(alt - self.target_altitude)
        is_upright = self._is_upright(ori)
        is_tumbling = self._is_tumbling(ang_vel)
        ang_vel_magnitude = np.linalg.norm(ang_vel)
        
        # Reward calculation
        reward = 0.0
        done = False
        info = {}
        
        # TUMBLE RECOVERY REWARDS
        if self.tumble_initiated and not self.tumble_recovered:
            reward -= ang_vel_magnitude * 10
            
            if is_upright:
                reward += 500
                if not self.tumble_recovery_started:
                    self.tumble_recovery_started = True
                    if self.debug:
                        print(f"   🎯 Getting upright...")
            else:
                reward -= 50
            
            if ang_vel_magnitude < 1.0:
                reward += 100
            
            reward -= dist_from_center * 0.5
            reward -= dist_from_target_alt * 0.5
        
        # AFTER RECOVERY: Normal hover rewards
        elif self.tumble_recovered or not self.will_tumble:
            if dist_from_center < 0.5:
                reward += 20
            else:
                reward -= dist_from_center * 2
            
            if dist_from_target_alt < 0.5:
                reward += 15
            else:
                reward -= dist_from_target_alt * 3
            
            if is_upright:
                reward += 10
                self.stable_steps += 1
            else:
                reward -= 20
                self.stable_steps = 0
            
            if ang_vel_magnitude < 0.2:
                reward += 5
            else:
                reward -= ang_vel_magnitude * 2
            
            if self.stable_steps > 50:
                reward += 10
        
        # Termination conditions
        if alt < 1.0:
            reward -= 1000
            done = True
            info['reason'] = 'crash'
        elif alt > 20.0:
            reward -= 500
            done = True
            info['reason'] = 'too_high'
        elif dist_from_center > 10.0:
            reward -= 500
            done = True
            info['reason'] = 'too_far'
        elif self.episode_steps >= self.max_steps:
            done = True
            info['reason'] = 'timeout'
        
        # Info for tracking
        info['altitude'] = alt
        info['distance'] = dist_from_center
        info['is_upright'] = is_upright
        info['is_tumbling'] = is_tumbling
        info['angular_velocity_mag'] = ang_vel_magnitude
        info['tumble_initiated'] = self.tumble_initiated
        info['tumble_recovered'] = self.tumble_recovered
        info['wind_magnitude'] = np.linalg.norm(self.current_wind)
        
        if self.tumble_recovered:
            info['recovery_steps'] = self.recovery_steps
        
        return obs, reward, done, False, info
    
    def close(self):
        """Cleanup"""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)