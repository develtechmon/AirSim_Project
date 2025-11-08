"""
DRONE FLIP RECOVERY ENVIRONMENT - WITH DISTURBANCE INJECTOR (13 OBSERVATIONS)
==============================================================================
Stage 3: Uses external DisturbanceInjector for realistic bird attacks and impacts

The drone must learn to:
1. Detect tumbling/disturbances (using angular velocity)
2. Apply counter-rotation to stop the tumble
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
import sys
import os

# Import the disturbance injector
sys.path.append(os.path.dirname(__file__))
from disturbance_injector import DisturbanceInjector, DisturbanceType


class DroneFlipRecoveryEnv(gym.Env):
    """
    Drone flip recovery with REALISTIC disturbance injection
    """
    
    def __init__(self, target_altitude=10.0, max_steps=500, 
                 wind_strength=5.0, flip_prob=0.9, debug=False):
        super().__init__()
        
        self.target_altitude = target_altitude
        self.max_steps = max_steps
        self.wind_strength = wind_strength
        self.disturbance_prob = flip_prob  # Probability of disturbance
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
        
        # Create disturbance injector
        self.disturbance_injector = DisturbanceInjector(self.client)
        
        # Episode tracking
        self.episode_steps = 0
        self.stable_steps = 0
        self.current_wind = np.zeros(3)
        self.wind_change_timer = 0
        self.wind_change_interval = np.random.uniform(1.0, 3.0)
        
        # Disturbance tracking
        self.disturbance_initiated = False
        self.disturbance_recovered = False
        self.disturbance_start_step = 0
        self.recovery_steps = 0
        self.disturbance_info = {}
        
        if self.debug:
            print(f"✓ Drone Flip Recovery Environment (DISTURBANCE INJECTOR)")
            print(f"  - Target: [0, 0, {-self.target_altitude}]")
            print(f"  - Max steps: {self.max_steps}")
            print(f"  - Wind strength: 0-{self.wind_strength} m/s")
            print(f"  - Disturbance probability: {self.disturbance_prob*100:.0f}%")
            print(f"  - Mode: EXTERNAL DISTURBANCE INJECTION")
    
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
        
        # Disturbance tracking
        self.disturbance_initiated = False
        self.disturbance_recovered = False
        self.disturbance_start_step = 0
        self.recovery_steps = 0
        self.disturbance_info = {}
        
        # Decide if disturbance will happen
        self.will_have_disturbance = np.random.random() < self.disturbance_prob
        if self.will_have_disturbance:
            self.disturbance_trigger_step = np.random.randint(20, 50)
            # Choose random disturbance type (focus on tumbling types)
            self.disturbance_type = np.random.choice([
                DisturbanceType.BIRD_ATTACK,
                DisturbanceType.FLIP,
                DisturbanceType.SPIN,
                #DisturbanceType.COLLISION,Disable collision for now
            ])
            if self.debug:
                print(f"   ⚠️  Disturbance scheduled: {self.disturbance_type.value} at step {self.disturbance_trigger_step}")
        else:
            self.disturbance_trigger_step = -1
            if self.debug:
                print(f"   ✅ No disturbance this episode")
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step"""
        self.episode_steps += 1
        
        # Trigger disturbance if scheduled
        if self.will_have_disturbance and not self.disturbance_initiated:
            if self.episode_steps >= self.disturbance_trigger_step:
                # Inject disturbance!
                intensity = np.random.uniform(0.8, 1.5)  # High intensity
                self.disturbance_info = self.disturbance_injector.inject_disturbance(
                    self.disturbance_type,
                    intensity=intensity
                )
                self.disturbance_initiated = True
                self.disturbance_start_step = self.episode_steps
                
                if self.debug:
                    print(f"   🐦 DISTURBANCE APPLIED!")
                    print(f"      Type: {self.disturbance_info['type']}")
                    print(f"      Intensity: {intensity:.2f}")
                    if 'angular_velocity' in self.disturbance_info:
                        print(f"      Angular velocity: {self.disturbance_info['angular_velocity']:.1f} deg/s")
        
        # Execute action (velocity command)
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
        
        # Check for recovery
        if self.disturbance_initiated and not self.disturbance_recovered:
            # Consider recovered if stable and upright
            if ang_vel_magnitude < 0.8 and is_upright and dist_from_target_alt < 2.0:
                self.disturbance_recovered = True
                self.recovery_steps = self.episode_steps - self.disturbance_start_step
                if self.debug:
                    print(f"   ✅ RECOVERED! Took {self.recovery_steps} steps ({self.recovery_steps * 0.05:.1f}s)")
        
        # Reward calculation
        reward = 0.0
        done = False
        info = {}
        
        # DISTURBANCE RECOVERY REWARDS
        if self.disturbance_initiated and not self.disturbance_recovered:
            # During disturbance - reward for reducing angular velocity
            reward -= ang_vel_magnitude * 10
            
            if is_upright:
                reward += 500
            else:
                reward -= 50
            
            if ang_vel_magnitude < 1.0:
                reward += 100
            
            # Altitude maintenance
            if alt > 2.0:  # Still flying
                reward += 50
            
            reward -= dist_from_center * 0.5
            reward -= dist_from_target_alt * 0.5
        
        # AFTER RECOVERY: Normal hover rewards
        elif self.disturbance_recovered or not self.will_have_disturbance:
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
        info['tumble_initiated'] = self.disturbance_initiated
        info['tumble_recovered'] = self.disturbance_recovered
        info['wind_magnitude'] = np.linalg.norm(self.current_wind)
        
        if self.disturbance_recovered:
            info['recovery_steps'] = self.recovery_steps
        
        return obs, reward, done, False, info
    
    def close(self):
        """Cleanup"""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)