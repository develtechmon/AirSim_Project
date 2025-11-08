"""
DRONE FLIP RECOVERY ENVIRONMENT (13 OBSERVATIONS)
==================================================
Stage 3: Combines flip recovery + wind disturbances + hover control

The drone must learn to:
1. Detect when it's flipped (using orientation quaternion)
2. Execute recovery maneuvers (using angular velocity)
3. Stabilize after recovery
4. Handle wind disturbances during recovery
5. Return to hover position

Observation Space: (13,)
    - position: [x, y, z]
    - velocity: [vx, vy, vz]  
    - orientation: [qw, qx, qy, qz]
    - angular_velocity: [wx, wy, wz] ← CRITICAL for flip detection!

Action Space: (3,)
    - velocity commands: [vx, vy, vz] in range [-5, 5] m/s

This is the FINAL stage - the ultimate test!
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time


class DroneFlipRecoveryEnv(gym.Env):
    """
    Drone flip recovery environment with wind disturbances
    
    The drone starts either:
    - Upright (normal hover)
    - Flipped/tilted at random angles
    
    Must recover orientation and return to stable hover.
    """
    
    def __init__(self, target_altitude=10.0, max_steps=500, 
                 wind_strength=5.0, flip_prob=0.5, debug=False):
        super().__init__()
        
        self.target_altitude = target_altitude
        self.max_steps = max_steps
        self.wind_strength = wind_strength
        self.flip_prob = flip_prob  # Probability of starting flipped
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
        self.started_flipped = False
        self.has_recovered = False
        self.recovery_steps = 0
        
        if self.debug:
            print(f"✓ Drone Flip Recovery Environment (13 observations)")
            print(f"  - Target: [0, 0, {-self.target_altitude}]")
            print(f"  - Action: 3D velocity [-5, 5] m/s")
            print(f"  - Max steps: {self.max_steps}")
            print(f"  - Wind strength: 0-{self.wind_strength} m/s")
            print(f"  - Flip probability: {self.flip_prob*100:.0f}%")
            print(f"  - Observations: 13 (pos + vel + ori + ang_vel)")
    
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
        """Check if drone is upright (not flipped)"""
        # Convert quaternion to up vector
        # For upright: up vector should point mostly upward (z > 0.7)
        qw, qx, qy, qz = orientation
        
        # Up vector in world frame
        up_x = 2 * (qx * qz - qw * qy)
        up_y = 2 * (qy * qz + qw * qx)
        up_z = 1 - 2 * (qx * qx + qy * qy)
        
        # If z-component of up vector > 0.7, drone is mostly upright
        return up_z > 0.7
    
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
        
        # Decide if starting flipped
        self.started_flipped = np.random.random() < self.flip_prob
        
        if self.started_flipped:
            # Apply random rotation (flip)
            # Random angles for pitch, roll, yaw
            pitch = np.random.uniform(-np.pi, np.pi)  # Can be fully inverted
            roll = np.random.uniform(-np.pi, np.pi)
            yaw = np.random.uniform(-np.pi, np.pi)
            
            # Set pose with random orientation
            pose = self.client.simGetVehiclePose()
            pose.position = airsim.Vector3r(start_x, start_y, -self.target_altitude)
            pose.orientation = airsim.to_quaternion(pitch, roll, yaw)
            self.client.simSetVehiclePose(pose, True)
            time.sleep(0.5)
            
            if self.debug:
                print(f"   🔄 Started FLIPPED! (pitch={np.degrees(pitch):.0f}°, roll={np.degrees(roll):.0f}°)")
        else:
            if self.debug:
                print(f"   ✅ Started upright")
        
        # Initialize wind
        self.current_wind = self._get_wind()
        self._apply_wind()
        
        # Reset tracking
        self.episode_steps = 0
        self.stable_steps = 0
        self.wind_change_timer = 0
        self.wind_change_interval = np.random.uniform(1.0, 3.0)
        self.has_recovered = False
        self.recovery_steps = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action"""
        self.episode_steps += 1
        self.wind_change_timer += 0.05
        
        # Change wind periodically
        if self.wind_change_timer >= self.wind_change_interval:
            self.current_wind = self._get_wind()
            self._apply_wind()
            self.wind_change_timer = 0
            self.wind_change_interval = np.random.uniform(1.0, 3.0)
        
        # Apply velocity action
        vx, vy, vz = action
        self.client.moveByVelocityAsync(
            float(vx), float(vy), float(vz),
            duration=0.05,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        ).join()
        
        # Get new state
        obs = self._get_observation()
        drone_state = self.client.getMultirotorState()
        pos = drone_state.kinematics_estimated.position
        ori = drone_state.kinematics_estimated.orientation
        
        # Extract orientation for upright check
        orientation = np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])
        is_upright = self._is_upright(orientation)
        
        # Track recovery
        if self.started_flipped and not self.has_recovered:
            if is_upright:
                self.has_recovered = True
                self.recovery_steps = self.episode_steps
                if self.debug:
                    print(f"   ✅ RECOVERED at step {self.recovery_steps}!")
        
        # Calculate metrics
        altitude = -pos.z_val
        dist_from_center = np.sqrt(pos.x_val**2 + pos.y_val**2)
        dist_from_target_alt = abs(altitude - self.target_altitude)
        
        # Check if stable
        is_stable = (dist_from_center < 0.5) and (dist_from_target_alt < 0.5) and is_upright
        if is_stable:
            self.stable_steps += 1
        
        # Reward calculation
        reward = 0
        
        # FLIP RECOVERY REWARDS (HIGHEST PRIORITY)
        if self.started_flipped:
            if not self.has_recovered:
                # Not yet recovered - huge bonus for getting upright
                if is_upright:
                    reward += 500  # MASSIVE reward for recovering
                else:
                    # Small penalty for staying flipped (encourage recovery)
                    reward -= 5
            else:
                # Already recovered - now just maintain hover
                # Same rewards as Stage 2
                pass
        
        # HOVER REWARDS (apply to all, or after recovery)
        if not self.started_flipped or self.has_recovered:
            # Distance rewards
            if dist_from_center < 0.5:
                reward += 20
            else:
                reward -= dist_from_center * 2
            
            if dist_from_target_alt < 0.5:
                reward += 20
            else:
                reward -= dist_from_target_alt * 3
            
            # Hovering bonus
            if is_stable:
                reward += 50
            
            # Action penalty (smooth control)
            action_magnitude = np.linalg.norm(action)
            reward -= action_magnitude * 2
            
            # Wind compensation bonus
            wind_magnitude = np.linalg.norm(self.current_wind)
            if is_stable and wind_magnitude > 2.0:
                reward += wind_magnitude * 10
        
        # Penalty for being upside down
        if not is_upright:
            reward -= 10
        
        # Termination conditions
        terminated = False
        truncated = False
        reason = "running"
        
        # Out of bounds
        if dist_from_center > 20:
            terminated = True
            reward -= 500
            reason = "out_of_bounds"
        
        # Altitude violation
        if altitude < 2 or altitude > 30:
            terminated = True
            reward -= 500
            reason = "altitude_violation"
        
        # Collision
        collision = self.client.simGetCollisionInfo()
        if collision.has_collided:
            terminated = True
            reward -= 1000
            reason = "collision"
        
        # Time limit
        if self.episode_steps >= self.max_steps:
            truncated = True
            reason = "timeout"
        
        # Debug output
        if self.debug and (self.episode_steps % 50 == 0 or terminated or truncated):
            upright_str = "✅" if is_upright else "🔄"
            print(f"[Step {self.episode_steps}] {upright_str} Alt={altitude:.1f}m | "
                  f"Dist={dist_from_center:.2f}m | "
                  f"Wind={np.linalg.norm(self.current_wind):.1f}m/s | "
                  f"Reward={reward:.1f}")
            if terminated or truncated:
                print(f"   ❌ {reason}")
        
        info = {
            'altitude': altitude,
            'distance': dist_from_center,
            'stable_steps': self.stable_steps,
            'reason': reason,
            'wind_magnitude': np.linalg.norm(self.current_wind),
            'is_flipped': not is_upright,
            'has_recovered': self.has_recovered,
            'recovery_steps': self.recovery_steps,
            'started_flipped': self.started_flipped
        }
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Cleanup"""
        try:
            # Clear wind
            self.client.simSetWind(airsim.Vector3r(0, 0, 0))
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass


if __name__ == "__main__":
    # Test the environment
    print("Testing flip recovery environment (13 observations)...")
    
    env = DroneFlipRecoveryEnv(flip_prob=1.0, debug=True)  # Always start flipped for test
    
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Expected: (13,) - Got: {obs.shape}")
    
    if obs.shape[0] != 13:
        print(f"❌ ERROR: Expected 13 observations, got {obs.shape[0]}!")
    else:
        print("✅ Observation space correct: 13 dimensions")
    
    print(f"Observation: {obs}")
    print(f"Position: [{obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f}]")
    print(f"Velocity: [{obs[3]:.2f}, {obs[4]:.2f}, {obs[5]:.2f}]")
    print(f"Orientation: [{obs[6]:.2f}, {obs[7]:.2f}, {obs[8]:.2f}, {obs[9]:.2f}]")
    print(f"Angular Vel: [{obs[10]:.2f}, {obs[11]:.2f}, {obs[12]:.2f}]")
    
    print("\nRunning 100 random actions to test flip scenario...")
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get('has_recovered'):
            print(f"\n✅ Recovery detected at step {i}!")
        
        if terminated or truncated:
            print(f"\nEpisode ended: {info['reason']}")
            print(f"Started flipped: {info['started_flipped']}")
            print(f"Recovered: {info['has_recovered']}")
            if info['has_recovered']:
                print(f"Recovery time: {info['recovery_steps']} steps")
            break
    
    env.close()
    print("\n✅ Environment test complete!")
