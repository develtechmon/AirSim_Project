"""
PERFORMANCE-GATED CURRICULUM ENVIRONMENT
========================================
Only moves to next difficulty level when current level is MASTERED!

Key Features:
- Easy (0.7-0.9):   Must achieve 80% recovery before advancing
- Medium (0.9-1.1): Must achieve 70% recovery before advancing  
- Hard (1.1-1.5):   Train until 60% recovery (PhD target)

This ensures SOLID foundation before increasing difficulty!
"""

"""
PERFORMANCE-GATED CURRICULUM ENVIRONMENT - FIXED ALTITUDE RECOVERY
===================================================================
IMPROVEMENTS:
✅ Aggressive vertical velocity rewards (climb faster!)
✅ Progressive altitude urgency (closer to target = more reward)
✅ Balanced orientation vs altitude priorities
✅ Penalize wrong-direction vertical movement

This version makes the drone return to 30m altitude MUCH FASTER after recovery!


Commands : 
Option 2: Train new model from scratch with fixed rewards
python train_stage_3_gated_curriculum.py --timesteps 600000

Option 3: Continue training existing model (adapts to new rewards)
python stage_3/train_stage_3_gated_curriculum.py --stage2-model ./models/stage3_checkpoints/gated_curriculum_policy.zip --timesteps 200000 --lr 1e-5 (i'm using this)
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import sys
import os
from collections import deque

sys.path.append(os.path.dirname(__file__))
from disturbance_injector import DisturbanceInjector, DisturbanceType


class DroneFlipRecoveryEnvGated(gym.Env):
    """
    Performance-gated curriculum - only advance when mastered!
    FIXED: Fast altitude recovery after disturbance
    """
    
    def __init__(self, target_altitude=30.0, max_steps=600, 
                 wind_strength=5.0, flip_prob=0.9, debug=False):
        super().__init__()
        
        self.target_altitude = target_altitude
        self.max_steps = max_steps
        self.wind_strength = wind_strength
        self.disturbance_prob = flip_prob
        self.debug = debug
        
        # Action and observation spaces
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.disturbance_injector = DisturbanceInjector(self.client)
        
        # Episode tracking
        self.episode_steps = 0
        self.episode_count = 0
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
        self.previous_ang_vel_magnitude = 0
        
        # PERFORMANCE-GATED CURRICULUM STATE
        self.curriculum_level = 0
        self.level_names = ["EASY", "MEDIUM", "HARD"]
        self.recent_recoveries = deque(maxlen=50)
        self.advancement_thresholds = {
            0: 0.80,
            1: 0.70,
            2: 0.60
        }
        self.min_episodes_per_level = 50
        self.episodes_at_current_level = 0
        self.level_advanced = False
        self.advancement_info = {}
        
        if self.debug:
            print(f"✓ PERFORMANCE-GATED Curriculum Environment (FIXED ALTITUDE RECOVERY)")
            print(f"  - Starting level: {self.level_names[self.curriculum_level]}")
            print(f"  - Altitude: {self.target_altitude}m")
            print(f"  - Strategy: Fast vertical recovery after disturbance")
    
    def _get_intensity_for_level(self):
        """Get intensity range based on current curriculum level"""
        if self.curriculum_level == 0:
            return np.random.uniform(0.7, 0.9)
        elif self.curriculum_level == 1:
            return np.random.uniform(0.9, 1.1)
        else:
            return np.random.uniform(1.1, 1.5)
    
    def _check_curriculum_advancement(self):
        """Check if we should advance to next difficulty level"""
        if self.episodes_at_current_level < self.min_episodes_per_level:
            return
        
        if self.curriculum_level >= 2:
            return
        
        if len(self.recent_recoveries) < 30:
            return
        
        recovery_rate = np.mean(list(self.recent_recoveries))
        threshold = self.advancement_thresholds[self.curriculum_level]
        
        if recovery_rate >= threshold:
            old_level = self.curriculum_level
            self.curriculum_level += 1
            self.episodes_at_current_level = 0
            self.recent_recoveries.clear()
            
            self.level_advanced = True
            self.advancement_info = {
                'old_level': old_level,
                'new_level': self.curriculum_level,
                'recovery_rate': recovery_rate,
                'threshold': threshold,
                'episode': self.episode_count
            }
            
            if self.debug:
                print("\n" + "="*70)
                print("🎓 CURRICULUM ADVANCEMENT!")
                print("="*70)
                print(f"   Level {old_level} ({self.level_names[old_level]}) MASTERED!")
                print(f"   Recovery rate: {recovery_rate*100:.1f}% (needed {threshold*100:.0f}%)")
                print(f"   Advancing to Level {self.curriculum_level} ({self.level_names[self.curriculum_level]})")
                print(f"   💾 MODEL WILL BE AUTO-SAVED!")
                print("="*70 + "\n")
    
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
        self.previous_ang_vel_magnitude = 0
        
        # Decide if disturbance will happen
        self.will_have_disturbance = np.random.random() < self.disturbance_prob
        if self.will_have_disturbance:
            self.disturbance_trigger_step = np.random.randint(20, 50)
            self.disturbance_type = np.random.choice([
                DisturbanceType.BIRD_ATTACK,
                DisturbanceType.FLIP,
                DisturbanceType.SPIN,
            ])
            if self.debug:
                print(f"   ⚠️  Disturbance scheduled: {self.disturbance_type.value} at step {self.disturbance_trigger_step}")
                print(f"      Level {self.curriculum_level} ({self.level_names[self.curriculum_level]})")
                print(f"      Episodes at level: {self.episodes_at_current_level}")
                if len(self.recent_recoveries) >= 10:
                    print(f"      Recent recovery: {np.mean(list(self.recent_recoveries))*100:.0f}%")
        else:
            self.disturbance_trigger_step = -1
            if self.debug:
                print(f"   ✅ No disturbance this episode")
                print(f"      Level {self.curriculum_level} ({self.level_names[self.curriculum_level]})")
                print(f"      Episodes at level: {self.episodes_at_current_level}")
        
        self.episode_count += 1
        self.episodes_at_current_level += 1
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step"""
        self.episode_steps += 1
        
        # Trigger disturbance with GATED CURRICULUM INTENSITY
        if self.will_have_disturbance and not self.disturbance_initiated:
            if self.episode_steps >= self.disturbance_trigger_step:
                
                intensity = self._get_intensity_for_level()
                
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
                    print(f"      Curriculum Level: {self.curriculum_level} ({self.level_names[self.curriculum_level]})")
                    if 'angular_velocity' in self.disturbance_info:
                        print(f"      Angular velocity: {self.disturbance_info['angular_velocity']:.1f} deg/s")
        
        # Execute action
        action = np.clip(action, -5.0, 5.0)
        self.client.moveByVelocityAsync(
            float(action[0]),
            float(action[1]),
            float(action[2]),
            duration=0.05,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        ).join()
        
        # Update wind
        self.wind_change_timer += 1
        if self.wind_change_timer >= int(self.wind_change_interval / 0.05):
            self.current_wind = self._get_wind()
            self._apply_wind()
            self.wind_change_timer = 0
            self.wind_change_interval = np.random.uniform(1.0, 3.0)
        
        # Get observation
        obs = self._get_observation()
        pos, vel, ori, ang_vel = obs[0:3], obs[3:6], obs[6:10], obs[10:13]
        
        # Calculate metrics
        alt = -pos[2]
        dist_from_center = np.linalg.norm(pos[0:2])
        altitude_error = abs(alt - self.target_altitude)
        is_upright = self._is_upright(ori)
        is_tumbling = self._is_tumbling(ang_vel)
        ang_vel_magnitude = np.linalg.norm(ang_vel)
        
        # Check for recovery
        if self.disturbance_initiated and not self.disturbance_recovered:
            is_controlled = (ang_vel_magnitude < 1.2 and is_upright)
            is_safe_altitude = (alt > 3.0)
            horizontal_vel = np.linalg.norm(vel[0:2])
            is_stable = (horizontal_vel < 3.0)
            
            if is_controlled and is_safe_altitude and is_stable:
                self.disturbance_recovered = True
                self.recovery_steps = self.episode_steps - self.disturbance_start_step
                if self.debug:
                    print(f"   ✅ RECOVERED! Took {self.recovery_steps} steps ({self.recovery_steps * 0.05:.1f}s)")
                    print(f"      Angular velocity: {ang_vel_magnitude:.2f} rad/s (< 1.2)")
                    print(f"      Altitude: {alt:.1f}m (> 3.0m - SAFE)")
                    print(f"      Horizontal drift: {horizontal_vel:.2f} m/s (< 3.0)")
                    print(f"      Level: {self.curriculum_level} ({self.level_names[self.curriculum_level]})")
                    print(f"      Intensity: {self.disturbance_info.get('intensity', 1.0):.2f}x")
        
        # ================================================================
        # FIXED: THREE-PHASE RECOVERY WITH AGGRESSIVE ALTITUDE RECOVERY
        # ================================================================
        reward = 0.0
        done = False
        info = {}
        
        if self.disturbance_initiated and not self.disturbance_recovered:
            # Phase 1: EXTREME TUMBLING - Priority: Don't crash, stop spinning
            if ang_vel_magnitude > 4.0:
                reward -= ang_vel_magnitude * 35
                
                # Reward for slowing down tumbling
                if self.previous_ang_vel_magnitude > 0:
                    reduction = self.previous_ang_vel_magnitude - ang_vel_magnitude
                    if reduction > 0:
                        reward += reduction * 150
                
                # Bonus for getting below critical thresholds
                if ang_vel_magnitude < 5.0:
                    reward += 300
                if ang_vel_magnitude < 4.5:
                    reward += 200
                
                # ALTITUDE URGENCY - Don't crash!
                if alt > 20.0:
                    reward += 200
                elif alt > 15.0:
                    reward += 150
                elif alt > 10.0:
                    reward += 100
                elif alt > 5.0:
                    reward += 50
                elif alt > 2.0:
                    reward += 20
                else:
                    reward -= 300
            
            # Phase 2: MODERATE TUMBLING - Priority: Get upright AND climb
            elif ang_vel_magnitude > 1.0:
                reward -= ang_vel_magnitude * 20
                
                # Orientation rewards
                if is_upright:
                    reward += 1000
                else:
                    reward -= 200
                
                # Tumble reduction bonuses
                if ang_vel_magnitude < 3.0:
                    reward += 300
                if ang_vel_magnitude < 2.0:
                    reward += 400
                
                if self.previous_ang_vel_magnitude > 0:
                    reduction = self.previous_ang_vel_magnitude - ang_vel_magnitude
                    if reduction > 0:
                        reward += reduction * 100
                
                # ENHANCED ALTITUDE REWARDS - Encourage climbing!
                if alt > 25.0:
                    reward += 500  # ← INCREASED from 150
                elif alt > 20.0:
                    reward += 350  # ← INCREASED from 100
                elif alt > 15.0:
                    reward += 200  # ← INCREASED from 50
                elif alt > 10.0:
                    reward += 100
                elif alt > 5.0:
                    reward += 50
                elif alt > 2.0:
                    reward += 20
                else:
                    reward -= 200
                
                # 🆕 REWARD VERTICAL VELOCITY (climbing)
                vz = vel[2]  # Down is positive in NED
                if vz < -0.5:  # Climbing (negative = up)
                    reward += abs(vz) * 100  # ← Reward faster climbing!
                elif vz > 0.5:  # Descending
                    reward -= vz * 50  # ← Penalize descending
            
            # Phase 3: NEARLY STABLE - Priority: Return to target altitude FAST
            else:
                reward -= ang_vel_magnitude * 10
                
                # Orientation rewards (reduced since mostly stable)
                if is_upright:
                    reward += 800  # ← REDUCED from 1500
                else:
                    reward -= 150
                
                # Fine control bonuses
                if ang_vel_magnitude < 0.5:
                    reward += 500
                if ang_vel_magnitude < 0.3:
                    reward += 300
                
                # 🆕 AGGRESSIVE ALTITUDE RETURN REWARDS
                # Distance-based rewards (closer = better)
                if altitude_error < 1.0:
                    reward += 1000  # ← HUGE reward for being at target!
                elif altitude_error < 2.0:
                    reward += 800
                elif altitude_error < 3.0:
                    reward += 600
                elif altitude_error < 5.0:
                    reward += 400
                elif altitude_error < 8.0:
                    reward += 200
                else:
                    reward -= altitude_error * 20  # ← Penalize being far
                
                # 🆕 AGGRESSIVE VERTICAL VELOCITY REWARDS
                vz = vel[2]
                
                if alt < self.target_altitude - 2.0:
                    # Below target - CLIMB FAST!
                    if vz < -1.0:  # Climbing fast
                        reward += abs(vz) * 300  # ← BIG reward for fast climb
                    elif vz < -0.5:  # Climbing moderately
                        reward += abs(vz) * 150
                    elif vz < 0.0:  # Climbing slowly
                        reward += abs(vz) * 50
                    else:  # Descending while below target - BAD!
                        reward -= vz * 200
                
                elif alt > self.target_altitude + 2.0:
                    # Above target - descend moderately
                    if vz > 0.5:  # Descending
                        reward += vz * 100
                    elif vz < -0.5:  # Still climbing - wrong direction!
                        reward -= abs(vz) * 150
                
                else:
                    # Near target - slow down vertical velocity
                    if abs(vz) < 0.3:
                        reward += 200  # Reward stable hover
                    else:
                        reward -= abs(vz) * 50  # Penalize oscillation
                
                # Horizontal position (less important now)
                if dist_from_center < 1.0:
                    reward += 100
                elif dist_from_center > 3.0:
                    reward -= dist_from_center * 10
            
            # Global penalties (all phases)
            reward -= dist_from_center * 0.3
            reward -= altitude_error * 0.5  # ← INCREASED from 0.2
        
        # After recovery or no disturbance - maintain position
        elif self.disturbance_recovered or not self.will_have_disturbance:
            if dist_from_center < 0.5:
                reward += 20
            else:
                reward -= dist_from_center * 2
            if altitude_error < 0.5:
                reward += 15
            else:
                reward -= altitude_error * 3
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
        
        self.previous_ang_vel_magnitude = ang_vel_magnitude
        
        # Termination
        if alt < 0.5:
            reward -= 1500
            done = True
            info['reason'] = 'crash'
        elif alt > 40.0:
            reward -= 500
            done = True
            info['reason'] = 'too_high'
        elif dist_from_center > 15.0:
            reward -= 500
            done = True
            info['reason'] = 'too_far'
        elif self.episode_steps >= self.max_steps:
            done = True
            info['reason'] = 'timeout'
        
        # Info
        info['altitude'] = alt
        info['distance'] = dist_from_center
        info['is_upright'] = is_upright
        info['is_tumbling'] = is_tumbling
        info['angular_velocity_mag'] = ang_vel_magnitude
        info['tumble_initiated'] = self.disturbance_initiated
        info['tumble_recovered'] = self.disturbance_recovered
        info['wind_magnitude'] = np.linalg.norm(self.current_wind)
        info['curriculum_level'] = self.curriculum_level
        
        if self.disturbance_recovered:
            info['recovery_steps'] = self.recovery_steps
            self.recent_recoveries.append(1)
        elif self.disturbance_initiated and done:
            self.recent_recoveries.append(0)
        
        if self.disturbance_initiated:
            info['disturbance_intensity'] = self.disturbance_info.get('intensity', 1.0)
        
        # Check if should advance curriculum
        if done:
            self._check_curriculum_advancement()
        
        return obs, reward, done, False, info
    
    def close(self):
        """Cleanup"""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)


# Keep same class name for compatibility
DroneFlipRecoveryEnv = DroneFlipRecoveryEnvGated