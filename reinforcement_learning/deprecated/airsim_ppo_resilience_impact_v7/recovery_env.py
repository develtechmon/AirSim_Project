"""
Impact Recovery Environment - NO FLIPS VERSION
✅ Phase 0: PURE hovering (NO disturbances AT ALL)
✅ Disturbances ONLY in Phase 1+
✅ Actually learnable
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

from disturbance_injector import DisturbanceInjector, DisturbanceType


class RecoveryEnv(gym.Env):
    """
    SAFE progressive learning - no flips in Phase 0
    """
    
    def __init__(self,
                 target_altitude=-10.0,
                 crash_altitude=-0.5,
                 max_episode_steps=1200,
                 disturbance_frequency=400,
                 enable_tracking=False):
        
        super(RecoveryEnv, self).__init__()
        
        self.target_altitude = target_altitude
        self.crash_altitude = crash_altitude
        self.max_episode_steps = max_episode_steps
        self.disturbance_frequency = disturbance_frequency
        self.enable_tracking = enable_tracking
        
        # Recovery thresholds
        self.recovery_tilt_threshold = 35.0
        self.recovery_angular_vel_threshold = 60.0
        self.recovery_linear_vel_threshold = 6.0
        
        # Action space
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([-180, -180, -180, -500, -500, -500,
                         -50, -50, -50, -200, -200, -100, 0, 0], dtype=np.float32),
            high=np.array([180, 180, 180, 500, 500, 500,
                          50, 50, 50, 200, 200, 0, 10, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Connect to AirSim
        print("Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("✓ Connected to AirSim")
        
        # Disturbance injector
        self.disturbance_injector = DisturbanceInjector(self.client)
        
        # Tracking
        self.episode_steps = 0
        self.total_episodes = 0
        self.total_steps_trained = 0
        self.successful_episodes = 0
        self.total_crashes = 0
        
        # Episode state
        self.disturbance_count = 0
        self.recovery_count = 0
        self.steps_since_disturbance = 0
        self.currently_disturbed = False
        self.stable_steps = 0
        
        # Reward tracking
        self.cumulative_reward = 0
        self.best_tilt_this_episode = float('inf')
        
        # Progressive difficulty
        self.curriculum_phase = 0
        
        print("\n" + "="*70)
        print("✓ Recovery Environment - SAFE LEARNING")
        print("="*70)
        print("  Phase 0 (0-1000 eps): PURE HOVERING (NO disturbances)")
        print("  Phase 1 (1000-2500): Micro disturbances (0.02-0.08)")
        print("  Phase 2 (2500-4500): Tiny disturbances (0.08-0.20)")
        print("  Phase 3 (4500+): Small disturbances (0.20-0.50)")
        print(f"  Tracking: {'ENABLED' if enable_tracking else 'DISABLED'}")
        print("="*70 + "\n")
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Update phase
        self._update_curriculum_phase()
        
        # Reset AirSim
        try:
            self.client.reset()
            time.sleep(0.3)
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            
            # Take off
            self.client.takeoffAsync().join()
            time.sleep(0.5)
            
            # Move to hovering position
            self.client.moveToPositionAsync(0, 0, self.target_altitude, 5).join()
            time.sleep(1.0)  # Let it stabilize
            
        except Exception as e:
            print(f"⚠️  Reset error: {e}")
            self._reconnect_airsim()
        
        # Reset counters
        self.episode_steps = 0
        self.disturbance_count = 0
        self.recovery_count = 0
        self.steps_since_disturbance = 0
        self.currently_disturbed = False
        self.stable_steps = 0
        self.cumulative_reward = 0
        self.best_tilt_this_episode = float('inf')
        
        # ✅✅✅ CRITICAL: NO DISTURBANCE ON RESET! ✅✅✅
        # Disturbances only happen in step() function, never in reset()
        
        self.total_episodes += 1
        
        if self.enable_tracking:
            print(f"\n{'='*70}")
            print(f"EPISODE {self.total_episodes} START (Phase {self.curriculum_phase})")
            print(f"{'='*70}")
        
        obs = self._get_observation()
        info = {
            "episode": self.total_episodes,
            "phase": self.curriculum_phase,
            "disturbance_count": 0,
            "recovery_count": 0,
            "total_steps": self.total_steps_trained
        }
        
        return obs, info
    
    def step(self, action):
        """Execute timestep"""
        self.episode_steps += 1
        self.steps_since_disturbance += 1
        self.total_steps_trained += 1
        
        # Force termination at max steps
        if self.episode_steps >= self.max_episode_steps:
            obs = self._get_observation()
            
            if self.curriculum_phase == 0:
                success = True  # Phase 0: survived = success
            else:
                if self.disturbance_count > 0:
                    recovery_rate = self.recovery_count / self.disturbance_count
                    success = recovery_rate >= 0.5
                else:
                    success = True
            
            if success:
                self.successful_episodes += 1
            
            if self.enable_tracking:
                print(f"\n✓ Episode {self.total_episodes} complete")
                print(f"  Phase: {self.curriculum_phase}")
                print(f"  Best tilt: {self.best_tilt_this_episode:.1f}°")
                print(f"  Reward: {self.cumulative_reward:.0f}")
            
            info = {
                "episode_steps": self.episode_steps,
                "phase": self.curriculum_phase,
                "recovery_count": self.recovery_count,
                "disturbance_count": self.disturbance_count,
                "best_tilt": self.best_tilt_this_episode,
                "episode_success": success,
                "cumulative_reward": self.cumulative_reward,
                "total_steps": self.total_steps_trained
            }
            
            return obs, 0, success, not success, info
        
        # Validate action
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Apply action - GENTLE controls
        throttle = np.clip(float(action[0]), 0.3, 1.0)
        roll_rate = np.clip(float(action[1]) * 60, -60.0, 60.0)   # ✅ Very slow (was 80)
        pitch_rate = np.clip(float(action[2]) * 60, -60.0, 60.0)
        yaw_rate = np.clip(float(action[3]) * 60, -60.0, 60.0)
        
        try:
            self.client.moveByAngleRatesThrottleAsync(
                roll_rate=roll_rate,
                pitch_rate=pitch_rate,
                yaw_rate=yaw_rate,
                throttle=throttle,
                duration=0.1
            ).join()
        except Exception as e:
            print(f"⚠️  Action error: {e}")
            obs = self._get_safe_observation()
            return obs, -100, False, True, {"error": "action_failed"}
        
        # Get state
        try:
            state = self.client.getMultirotorState()
            roll, pitch, yaw = self._get_orientation_degrees(state)
            tilt = self._calculate_tilt(roll, pitch)
            altitude = state.kinematics_estimated.position.z_val
            angular_vel = self._get_angular_velocity_magnitude(state)
            linear_vel = self._get_linear_velocity_magnitude(state)
            
            if np.isnan(tilt) or np.isnan(altitude):
                raise ValueError("NaN")
                
        except Exception as e:
            print(f"⚠️  State error: {e}")
            obs = self._get_safe_observation()
            return obs, -100, False, True, {"error": "state_failed"}
        
        # Track best tilt
        if tilt < self.best_tilt_this_episode:
            self.best_tilt_this_episode = tilt
        
        # Check stability
        is_stable = (
            tilt < self.recovery_tilt_threshold and
            angular_vel < self.recovery_angular_vel_threshold and
            linear_vel < self.recovery_linear_vel_threshold
        )
        
        if is_stable:
            self.stable_steps += 1
        else:
            self.stable_steps = 0
        
        # Check recovery (30 steps = 3 seconds)
        just_recovered = False
        if self.currently_disturbed and self.stable_steps >= 30:
            self.currently_disturbed = False
            self.recovery_count += 1
            just_recovered = True
            
            if self.enable_tracking:
                print(f"\n✅ RECOVERY #{self.recovery_count} (Phase {self.curriculum_phase})")
        
        # ✅✅✅ INJECT DISTURBANCE - ONLY IN PHASE 1+ ✅✅✅
        if self.curriculum_phase >= 1:  # ✅ ONLY if Phase 1 or higher
            if (self.episode_steps >= 150 and  # ✅ Wait 15 seconds first
                self.steps_since_disturbance >= self.disturbance_frequency and
                not self.currently_disturbed):
                
                try:
                    self._apply_curriculum_disturbance()
                    self.currently_disturbed = True
                    self.steps_since_disturbance = 0
                    self.stable_steps = 0
                except Exception as e:
                    print(f"⚠️  Disturbance failed: {e}")
        
        # Check crash
        crashed = altitude >= self.crash_altitude
        
        if crashed:
            self.total_crashes += 1
            if self.enable_tracking:
                print(f"\n💥 CRASHED at altitude {altitude:.2f}m")
        
        terminated = False
        truncated = crashed
        
        # Calculate reward
        reward = self._calculate_reward(
            tilt, altitude, angular_vel, linear_vel,
            crashed, is_stable, just_recovered
        )
        
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        self.cumulative_reward += reward
        
        # Info
        info = {
            "episode_steps": self.episode_steps,
            "tilt": tilt,
            "altitude": altitude,
            "crashed": crashed,
            "phase": self.curriculum_phase,
            "recovery_count": self.recovery_count,
            "disturbance_count": self.disturbance_count,
            "best_tilt": self.best_tilt_this_episode,
            "cumulative_reward": self.cumulative_reward,
            "total_steps": self.total_steps_trained,
            "is_stable": is_stable
        }
        
        # Debug output
        if self.enable_tracking and (self.episode_steps % 300 == 0):
            print(f"[Ep {self.total_episodes}, Step {self.episode_steps}] "
                  f"Phase: {self.curriculum_phase}, Tilt: {tilt:.1f}°, Alt: {altitude:.1f}m")
        
        obs = self._get_observation()
        
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = self._get_safe_observation()
        
        return obs, reward, terminated, truncated, info
    
    def _update_curriculum_phase(self):
        """Update phase"""
        if self.total_episodes < 1000:
            self.curriculum_phase = 0
            
        elif self.total_episodes < 2500:
            if self.curriculum_phase == 0:
                print(f"\n{'='*70}")
                print(f"🎓 PHASE 1 UNLOCKED (Episode {self.total_episodes})")
                print(f"   Micro disturbances (intensity 0.02-0.08)")
                print(f"{'='*70}\n")
            self.curriculum_phase = 1
            
        elif self.total_episodes < 4500:
            if self.curriculum_phase == 1:
                print(f"\n{'='*70}")
                print(f"🎓 PHASE 2 UNLOCKED (Episode {self.total_episodes})")
                print(f"   Tiny disturbances (intensity 0.08-0.20)")
                print(f"{'='*70}\n")
            self.curriculum_phase = 2
            
        else:
            if self.curriculum_phase == 2:
                print(f"\n{'='*70}")
                print(f"🎓 PHASE 3 UNLOCKED (Episode {self.total_episodes})")
                print(f"   Small disturbances (intensity 0.20-0.50)")
                print(f"{'='*70}\n")
            self.curriculum_phase = 3
    
    def _apply_curriculum_disturbance(self):
        """Apply disturbance"""
        # ✅ This function is ONLY called when curriculum_phase >= 1
        
        if self.curriculum_phase == 1:
            # Micro disturbances
            disturbance_type = DisturbanceType.WIND_GUST
            intensity = np.random.uniform(0.02, 0.08)
            
        elif self.curriculum_phase == 2:
            # Tiny disturbances
            disturbance_type = np.random.choice([
                DisturbanceType.WIND_GUST,
                DisturbanceType.DROP
            ])
            intensity = np.random.uniform(0.08, 0.20)
            
        else:
            # Small disturbances
            disturbance_type = np.random.choice([
                DisturbanceType.WIND_GUST,
                DisturbanceType.DROP,
                DisturbanceType.COLLISION
            ])
            intensity = np.random.uniform(0.20, 0.50)
        
        self.disturbance_injector.inject_disturbance(disturbance_type, intensity)
        self.disturbance_count += 1
        
        if self.enable_tracking:
            print(f"\n💥 Disturbance #{self.disturbance_count} (Phase {self.curriculum_phase})")
            print(f"   Type: {disturbance_type.value}, Intensity: {intensity:.3f}")
    
    def _calculate_reward(self, tilt, altitude, angular_vel, linear_vel,
                         crashed, is_stable, just_recovered):
        """Calculate reward"""
        reward = 0.0
        
        # 1. Recovery bonus
        if just_recovered:
            if self.curriculum_phase == 1:
                reward += 5000.0
            elif self.curriculum_phase == 2:
                reward += 8000.0
            else:
                reward += 10000.0
            return reward
        
        # 2. Crash penalty
        if crashed:
            return -500.0
        
        # 3. Altitude reward
        distance_from_ground = abs(altitude)
        target_distance = abs(self.target_altitude)
        altitude_error = abs(distance_from_ground - target_distance)
        
        if altitude_error < 0.5:
            reward += 200.0
        elif altitude_error < 1.0:
            reward += 150.0
        elif altitude_error < 2.0:
            reward += 100.0
        elif altitude_error < 3.0:
            reward += 50.0
        else:
            reward += 10.0
        
        # Too low penalty
        if distance_from_ground < 1.5:
            reward -= 200.0
        elif distance_from_ground < 3.0:
            reward -= 50.0
        
        # 4. Tilt reward
        if self.curriculum_phase == 0:
            # Phase 0: Hovering - strict
            if tilt < 3:
                reward += 150.0
            elif tilt < 8:
                reward += 100.0
            elif tilt < 15:
                reward += 50.0
            elif tilt < 25:
                reward += 10.0
            else:
                reward -= 10.0
        else:
            # Phase 1+: Recovery - lenient
            if tilt < 10:
                reward += 200.0
            elif tilt < 20:
                reward += 150.0
            elif tilt < 30:
                reward += 100.0
            elif tilt < 45:
                reward += 60.0
            elif tilt < 60:
                reward += 30.0
            elif tilt < 90:
                reward += 15.0
            elif tilt < 120:
                reward += 5.0
            else:
                reward -= 5.0
        
        # 5. Angular velocity
        if angular_vel < 10:
            reward += 100.0
        elif angular_vel < 30:
            reward += 60.0
        elif angular_vel < 60:
            reward += 30.0
        elif angular_vel < 100:
            reward += 10.0
        else:
            reward -= 5.0
        
        # 6. Linear velocity
        if linear_vel < 1.0:
            reward += 80.0
        elif linear_vel < 3.0:
            reward += 50.0
        elif linear_vel < 6.0:
            reward += 20.0
        else:
            reward += 5.0
        
        # 7. Stability bonus
        if is_stable:
            reward += 150.0 + (self.stable_steps * 10.0)
        
        # 8. Survival bonus
        reward += 10.0
        
        return reward
    
    def _get_observation(self):
        """Get observation"""
        try:
            state = self.client.getMultirotorState()
            
            roll, pitch, yaw = self._get_orientation_degrees(state)
            
            ang_vel = state.kinematics_estimated.angular_velocity
            roll_rate = np.degrees(ang_vel.x_val)
            pitch_rate = np.degrees(ang_vel.y_val)
            yaw_rate = np.degrees(ang_vel.z_val)
            
            lin_vel = state.kinematics_estimated.linear_velocity
            vx, vy, vz = lin_vel.x_val, lin_vel.y_val, lin_vel.z_val
            
            pos = state.kinematics_estimated.position
            x, y, z = pos.x_val, pos.y_val, pos.z_val
            
            time_ratio = min(self.steps_since_disturbance / self.disturbance_frequency, 1.0)
            disturbed_flag = 1.0 if self.currently_disturbed else 0.0
            
            obs = np.array([
                roll, pitch, yaw,
                roll_rate, pitch_rate, yaw_rate,
                vx, vy, vz,
                x, y, z,
                time_ratio,
                disturbed_flag
            ], dtype=np.float32)
            
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
            
            return obs
        except:
            return self._get_safe_observation()
    
    def _get_safe_observation(self):
        """Safe observation"""
        return np.array([
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, -10.0,
            0.5, 0.0
        ], dtype=np.float32)
    
    def _get_orientation_degrees(self, state):
        """Get orientation"""
        orientation = state.kinematics_estimated.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation)
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    
    def _calculate_tilt(self, roll, pitch):
        """Calculate tilt"""
        return np.sqrt(roll**2 + pitch**2)
    
    def _get_angular_velocity_magnitude(self, state):
        """Get angular velocity magnitude"""
        ang_vel = state.kinematics_estimated.angular_velocity
        return np.sqrt(
            np.degrees(ang_vel.x_val)**2 +
            np.degrees(ang_vel.y_val)**2 +
            np.degrees(ang_vel.z_val)**2
        )
    
    def _get_linear_velocity_magnitude(self, state):
        """Get linear velocity magnitude"""
        lin_vel = state.kinematics_estimated.linear_velocity
        return np.sqrt(
            lin_vel.x_val**2 +
            lin_vel.y_val**2 +
            lin_vel.z_val**2
        )
    
    def _reconnect_airsim(self):
        """Reconnect"""
        try:
            try:
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
            except:
                pass
            
            time.sleep(1)
            
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            
            return True
        except:
            return False
    
    def close(self):
        """Cleanup"""
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass
        
        print(f"\n✓ Closed. Episodes: {self.total_episodes}, Steps: {self.total_steps_trained:,}")


# ═══════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING RECOVERY ENVIRONMENT")
    print("="*70)
    print("This test will run 1 episode in Phase 0")
    print("Expected: Drone hovers smoothly (NO flips, NO disturbances)")
    print("="*70 + "\n")
    
    env = RecoveryEnv(
        disturbance_frequency=400,
        max_episode_steps=1200,
        enable_tracking=True
    )
    
    obs, info = env.reset()
    print(f"✓ Episode started (Phase {info['phase']})")
    print(f"✓ In Phase 0 - NO disturbances will happen")
    print(f"✓ Drone should just hover peacefully\n")
    
    for i in range(1200):
        # Random action (just for testing)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"\n{'='*70}")
            print(f"EPISODE ENDED")
            print(f"{'='*70}")
            print(f"  Steps: {i+1}")
            print(f"  Phase: {info['phase']}")
            print(f"  Best tilt: {info['best_tilt']:.1f}°")
            print(f"  Reward: {info['cumulative_reward']:.0f}")
            print(f"  Crashed: {info.get('crashed', False)}")
            print(f"{'='*70}\n")
            break
    
    env.close()
    
    print("✅ TEST COMPLETE!")
    print("If you saw NO flips and the drone hovered, it's working!")