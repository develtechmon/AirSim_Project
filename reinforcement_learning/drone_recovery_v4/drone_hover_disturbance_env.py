"""
DRONE HOVER DISTURBANCE ENVIRONMENT
====================================
Stage 2: Adds random wind disturbances to the hover environment.

The drone must learn to compensate for:
- Random wind gusts (up to 5 m/s)
- Changing wind directions
- Sustained wind pressure

This builds on Stage 1 hover skills to create a more robust controller.
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time


class DroneHoverDisturbanceEnv(gym.Env):
    """
    Drone hover environment with wind disturbances
    
    Observation Space: (10,)
        - position: [x, y, z]
        - velocity: [vx, vy, vz]  
        - orientation: [qw, qx, qy, qz]
    
    Action Space: (3,)
        - velocity commands: [vx, vy, vz] in range [-5, 5] m/s
    
    Disturbances:
        - Random wind gusts (changes every 1-3 seconds)
        - Wind strength: 0-5 m/s in random directions
    """
    
    def __init__(self, target_altitude=10.0, max_steps=500, wind_strength=5.0, debug=False):
        super().__init__()
        
        self.target_altitude = target_altitude
        self.max_steps = max_steps
        self.wind_strength = wind_strength
        self.debug = debug
        
        # Action space: velocity commands
        self.action_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Observation space: position + velocity + orientation
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
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
        
        if self.debug:
            print(f"✓ Drone Hover Disturbance Environment")
            print(f"  - Target: [0, 0, {-self.target_altitude}]")
            print(f"  - Action: 3D velocity [-5, 5] m/s")
            print(f"  - Max steps: {self.max_steps}")
            print(f"  - Wind strength: 0-{self.wind_strength} m/s")
    
    def _get_wind(self):
        """Generate random wind vector"""
        # Random direction
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        
        # Random strength (0 to max)
        strength = np.random.uniform(0, self.wind_strength)
        
        # Convert to cartesian
        wind_x = strength * np.sin(phi) * np.cos(theta)
        wind_y = strength * np.sin(phi) * np.sin(theta)
        wind_z = strength * np.cos(phi) * 0.3  # Less vertical wind
        
        return np.array([wind_x, wind_y, wind_z])
    
    def _apply_wind(self):
        """Apply wind to the drone"""
        self.client.simSetWind(airsim.Vector3r(
            float(self.current_wind[0]),
            float(self.current_wind[1]),
            float(self.current_wind[2])
        ))
    
    def _get_observation(self):
        """Get current state"""
        drone_state = self.client.getMultirotorState()
        
        pos = drone_state.kinematics_estimated.position
        vel = drone_state.kinematics_estimated.linear_velocity
        ori = drone_state.kinematics_estimated.orientation
        
        obs = np.array([
            pos.x_val, pos.y_val, pos.z_val,
            vel.x_val, vel.y_val, vel.z_val,
            ori.w_val, ori.x_val, ori.y_val, ori.z_val
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
        
        # Move to starting position with small random offset
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
        
        if self.debug:
            print(f"\n🔄 RESET")
            print(f"   Start pos: [{start_x:.1f}, {start_y:.1f}, {-self.target_altitude:.1f}]")
            print(f"   Initial wind: [{self.current_wind[0]:.1f}, {self.current_wind[1]:.1f}, {self.current_wind[2]:.1f}] m/s")
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action"""
        self.episode_steps += 1
        self.wind_change_timer += 0.05
        
        # Change wind periodically
        if self.wind_change_timer >= self.wind_change_interval:
            old_wind = self.current_wind.copy()
            self.current_wind = self._get_wind()
            self._apply_wind()
            self.wind_change_timer = 0
            self.wind_change_interval = np.random.uniform(1.0, 3.0)
            
            if self.debug and self.episode_steps % 50 == 0:
                print(f"   🌬️  Wind changed: {old_wind} → {self.current_wind}")
        
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
        
        # Calculate metrics
        altitude = -pos.z_val
        dist_from_center = np.sqrt(pos.x_val**2 + pos.y_val**2)
        dist_from_target_alt = abs(altitude - self.target_altitude)
        
        # Check if stable
        is_stable = (dist_from_center < 0.5) and (dist_from_target_alt < 0.5)
        if is_stable:
            self.stable_steps += 1
        
        # Reward calculation
        reward = 0
        
        # Distance rewards
        if dist_from_center < 0.5:
            reward += 20  # Close to center
        else:
            reward -= dist_from_center * 2
        
        if dist_from_target_alt < 0.5:
            reward += 20  # Close to target altitude
        else:
            reward -= dist_from_target_alt * 3
        
        # Hovering bonus
        if is_stable:
            reward += 50
        
        # Action penalty (smooth control)
        action_magnitude = np.linalg.norm(action)
        reward -= action_magnitude * 2
        
        # Wind compensation bonus (staying stable despite wind)
        wind_magnitude = np.linalg.norm(self.current_wind)
        if is_stable and wind_magnitude > 2.0:
            reward += wind_magnitude * 10  # Bonus for stability in high wind
        
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
            print(f"[Step {self.episode_steps}] Alt={altitude:.1f}m | "
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
            'wind_magnitude': np.linalg.norm(self.current_wind)
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
    print("Testing disturbance environment...")
    
    env = DroneHoverDisturbanceEnv(debug=True)
    
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    
    print("\nRunning 50 random actions...")
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"\nEpisode ended: {info['reason']}")
            break
    
    env.close()
    print("\n✅ Environment test complete!")