"""
Drone Sphere Chasing Environment for AirSim
Continuous action space for smooth drone control
"""

import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import time


class DroneChaseEnv(gym.Env):
    """
    Custom Environment for training a drone to chase and hit a sphere
    
    Action Space: Continuous
        - vx: velocity in x direction [-5, 5] m/s
        - vy: velocity in y direction [-5, 5] m/s
        - vz: velocity in z direction [-5, 5] m/s
    
    Observation Space:
        - Relative position to target (3D): dx, dy, dz
        - Current velocity (3D): vx, vy, vz
        - Distance to target (1D): euclidean distance
        - Altitude difference (1D): how far drone is from target's altitude
    
    Reward Structure:
        + Large reward: Hit the target
        + Small reward: Reduce distance to target (encourages chasing)
        - Penalty: Increase distance to target (discourages running away)
        - Penalty: Large altitude difference from target
        - Penalty: Going beyond altitude limits
    """
    
    def __init__(self, 
                 max_altitude=50,
                 min_altitude=-50, 
                 hit_threshold=2.0,
                 max_distance=100,
                 spawn_radius=30):
        super(DroneChaseEnv, self).__init__()
        
        # Environment parameters
        self.max_altitude = max_altitude
        self.min_altitude = min_altitude
        self.hit_threshold = hit_threshold  # Distance to consider a "hit"
        self.max_distance = max_distance  # Max distance before penalty
        self.spawn_radius = spawn_radius  # Radius for random sphere spawning
        
        # Action space: continuous velocities [vx, vy, vz]
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0]),
            high=np.array([5.0, 5.0, 5.0]),
            dtype=np.float32
        )
        
        # Observation space: [dx, dy, dz, vx, vy, vz, distance, altitude_diff]
        self.observation_space = spaces.Box(
            low=np.array([-200, -200, -200, -10, -10, -10, 0, -100]),
            high=np.array([200, 200, 200, 10, 10, 10, 200, 100]),
            dtype=np.float32
        )
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # Tracking variables
        self.previous_distance = None
        self.target_position = None
        self.episode_steps = 0
        self.max_episode_steps = 500  # REDUCED from 1000 - end faster
        
        # NEW: Track if drone is making progress
        self.steps_without_progress = 0  # Count steps where distance doesn't decrease
        self.max_steps_without_progress = 20  # REDUCED from 50 - much stricter!
        self.previous_position = None  # Track drone's previous position
        
        print("✓ Drone Chase Environment Initialized")
        print(f"  - Hit threshold: {hit_threshold}m")
        print(f"  - Altitude limits: [{min_altitude}, {max_altitude}]")
        print(f"  - Max episode steps: {self.max_episode_steps}")
    
    def reset(self, seed=None, options=None):
        """Reset the environment and spawn new target"""
        super().reset(seed=seed)
        
        # Reset AirSim
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Take off to starting position
        self.client.takeoffAsync().join()
        time.sleep(0.5)
        
        # Move drone to starting position (origin)
        self.client.moveToPositionAsync(0, 0, -10, 5).join()
        
        # Clear any old visual markers (previous spheres)
        self.client.simFlushPersistentMarkers()
        
        # Spawn target sphere at random location
        self._spawn_target()
        
        # Reset tracking variables
        self.episode_steps = 0
        self.steps_without_progress = 0  # Reset inactivity counter
        drone_state = self.client.getMultirotorState()
        drone_pos = self._get_position(drone_state)
        self.previous_distance = self._calculate_distance(drone_pos, self.target_position)
        self.previous_position = drone_pos.copy()  # Initialize previous position
        
        # Get initial observation
        obs = self._get_observation()
        info = {"episode_steps": self.episode_steps}
        
        return obs, info
    
    def step(self, action):
        """Execute one time step within the environment"""
        self.episode_steps += 1
        
        # Apply action: move drone with given velocities
        vx, vy, vz = action
        duration = 1.0  # Execute action for 1.0 second (was too short at 0.5!)
        
        # Calculate action magnitude - reward for TRYING to move
        action_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # CRITICAL: Use moveByVelocityZBodyFrame for more responsive control
        # This makes the drone actually MOVE instead of just hovering
        self.client.moveByVelocityAsync(
            float(vx), float(vy), float(vz), 
            duration=duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        ).join()
        
        # Get current state
        drone_state = self.client.getMultirotorState()
        drone_pos = self._get_position(drone_state)
        
        # Calculate metrics
        current_distance = self._calculate_distance(drone_pos, self.target_position)
        altitude_diff = abs(drone_pos[2] - self.target_position[2])
        
        # ═══════════════════════════════════════════════════════════
        # INACTIVITY DETECTION - Check if drone is actually trying
        # ═══════════════════════════════════════════════════════════
        if self.previous_position is not None:
            # Calculate how much the drone actually moved
            drone_movement = self._calculate_distance(drone_pos, self.previous_position)
            
            # If drone barely moved (< 0.5m in 1 second), it's being lazy!
            if drone_movement < 0.5:
                self.steps_without_progress += 1
            else:
                # Drone is moving, reset counter
                self.steps_without_progress = 0
        
        self.previous_position = drone_pos.copy()
        
        # Check if drone is stuck/inactive for too long
        is_inactive = self.steps_without_progress >= self.max_steps_without_progress
        
        # Check if drone hit the target
        hit_target = current_distance <= self.hit_threshold
        
        # Check if drone went out of bounds
        out_of_bounds = (
            drone_pos[2] > self.max_altitude or 
            drone_pos[2] < self.min_altitude or
            current_distance > self.max_distance
        )
        
        # Calculate reward
        reward = self._calculate_reward(
            current_distance=current_distance,
            previous_distance=self.previous_distance,
            altitude_diff=altitude_diff,
            hit_target=hit_target,
            out_of_bounds=out_of_bounds,
            drone_pos=drone_pos,
            action_magnitude=action_magnitude  # NEW: reward for trying!
        )
        
        # Additional penalty for being inactive
        if is_inactive:
            reward -= 50.0  # Big penalty for doing nothing!
            print(f"⚠️  INACTIVE DRONE! No significant movement for {self.max_steps_without_progress} steps")
        
        # Update tracking
        self.previous_distance = current_distance
        
        # ═══════════════════════════════════════════════════════════
        # TERMINATION CONDITIONS - When episode should end
        # ═══════════════════════════════════════════════════════════
        terminated = hit_target  # Episode ends successfully when target is hit
        
        truncated = (
            out_of_bounds or                              # Went out of bounds
            is_inactive or                                # Drone is inactive/stuck
            self.episode_steps >= self.max_episode_steps  # Max steps reached
        )
        
        # If hit target, respawn for continuous training within episode
        if hit_target:
            print(f"🎯 Target HIT at step {self.episode_steps}! Respawning...")
            self._spawn_target()
            self.previous_distance = self._calculate_distance(drone_pos, self.target_position)
            self.steps_without_progress = 0  # Reset inactivity counter
            self.previous_position = drone_pos.copy()  # Update previous position
        
        # Get observation
        obs = self._get_observation()
        
        info = {
            "episode_steps": self.episode_steps,
            "distance_to_target": current_distance,
            "hit_target": hit_target,
            "out_of_bounds": out_of_bounds,
            "altitude_diff": altitude_diff,
            "is_inactive": is_inactive,
            "steps_without_progress": self.steps_without_progress
        }
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self, current_distance, previous_distance, 
                          altitude_diff, hit_target, out_of_bounds, drone_pos,
                          action_magnitude):
        """
        NUCLEAR VERSION - EXTREME penalties for inaction!
        This WILL force the drone to move or die trying.
        """
        reward = 0.0
        
        # MAJOR REWARD: Hit the target! 🎉
        if hit_target:
            reward += 500.0  # MASSIVE reward (was 200)
            return reward
        
        # CRITICAL PENALTY: Out of bounds
        if out_of_bounds:
            reward -= 200.0  # Doubled penalty
            return reward
        
        # ═══════════════════════════════════════════════════════════
        # NEW: Reward for ACTION itself (trying to move)
        # ═══════════════════════════════════════════════════════════
        if action_magnitude > 1.0:  # Sent meaningful velocity command
            reward += action_magnitude * 3.0  # Reward just for TRYING
        elif action_magnitude < 0.5:  # Nearly zero action
            reward -= 20.0  # Penalty for lazy action
        
        # ═══════════════════════════════════════════════════════════
        # NUCLEAR OPTION: Distance change
        # ═══════════════════════════════════════════════════════════
        distance_delta = previous_distance - current_distance
        
        if distance_delta > 0:
            # EXCELLENT! Drone got closer - HUGE REWARD
            reward += distance_delta * 20.0  # Was 5.0, now 4x stronger!
        elif distance_delta < 0:
            # BAD! Drone moved away - CRUSHING PENALTY
            reward -= abs(distance_delta) * 30.0  # Was 8.0, now almost 4x
        else:
            # DRONE DIDN'T MOVE AT ALL - NUCLEAR PENALTY!
            reward -= 50.0  # Was 10.0, now 5x stronger!
        
        # ═══════════════════════════════════════════════════════════
        # ALTITUDE PENALTIES
        # ═══════════════════════════════════════════════════════════
        if altitude_diff > 3.0:  # Stricter threshold
            reward -= altitude_diff * 2.0  # Doubled
        
        if altitude_diff > 10.0:
            reward -= (altitude_diff - 10.0) * 5.0  # Much harsher
        
        # ═══════════════════════════════════════════════════════════
        # DISTANCE PENALTIES - Being far is painful
        # ═══════════════════════════════════════════════════════════
        if current_distance > 40:
            reward -= (current_distance - 40) * 2.0  # Severe far penalty
        
        reward -= current_distance * 0.2  # Was 0.05, now 4x
        
        # ═══════════════════════════════════════════════════════════
        # TIME PENALTY - Every second counts
        # ═══════════════════════════════════════════════════════════
        reward -= 2.0  # Was 0.5, now 4x stronger
        
        return reward
    
    def _get_observation(self):
        """Get current observation of the environment"""
        drone_state = self.client.getMultirotorState()
        drone_pos = self._get_position(drone_state)
        drone_vel = self._get_velocity(drone_state)
        
        # Relative position to target
        dx = self.target_position[0] - drone_pos[0]
        dy = self.target_position[1] - drone_pos[1]
        dz = self.target_position[2] - drone_pos[2]
        
        # Current velocity
        vx, vy, vz = drone_vel
        
        # Distance and altitude difference
        distance = self._calculate_distance(drone_pos, self.target_position)
        altitude_diff = dz  # Positive means target is above
        
        obs = np.array([dx, dy, dz, vx, vy, vz, distance, altitude_diff], 
                       dtype=np.float32)
        
        return obs
    
    def _create_point_cloud_sphere(self, position, radius=2.0, color=[1.0, 0.0, 0.0, 1.0]):
        """
        Create a visual sphere made of points in AirSim.
        Uses Fibonacci sphere algorithm for even point distribution.
        
        Think of it like creating a Christmas ornament from dots of light!
        """
        points = []
        num_points = 150  # More points = denser sphere
        
        # Fibonacci sphere algorithm - mathematically perfect point distribution
        for i in range(num_points):
            phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
            y_offset = 1 - (i / float(num_points - 1)) * 2  # Range: -1 to 1
            radius_at_y = np.sqrt(1 - y_offset * y_offset)  # Circular cross-section
            theta = phi * i
            
            # Calculate 3D coordinates
            x = np.cos(theta) * radius_at_y * radius
            y = np.sin(theta) * radius_at_y * radius
            z = y_offset * radius
            
            # Create point in AirSim coordinate system
            point = airsim.Vector3r(
                position[0] + x,
                position[1] + y,
                position[2] + z
            )
            points.append(point)
        
        # Plot the sphere in AirSim
        self.client.simPlotPoints(
            points,
            color_rgba=color,
            size=15.0,  # Size of each dot
            duration=120.0,  # How long it stays visible (seconds)
            is_persistent=True  # Stays visible even if script crashes
        )
    
    def _spawn_target(self):
        """Spawn target sphere at random location with visual representation"""
        # Random position within spawn radius
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(10, self.spawn_radius)
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = np.random.uniform(-30, -10)  # Random altitude
        
        self.target_position = np.array([x, y, z], dtype=np.float32)
        
        print(f"🎯 Target spawned at: ({x:.1f}, {y:.1f}, {z:.1f})")
        
        # Create visual point cloud sphere in AirSim
        # Red color for the target - easy to spot!
        self._create_point_cloud_sphere(
            position=[x, y, z],
            radius=2.0,
            color=[1.0, 0.0, 0.0, 1.0]  # RGBA: Red, full opacity
        )
    
    def _get_position(self, drone_state):
        """Extract position from drone state"""
        pos = drone_state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
    
    def _get_velocity(self, drone_state):
        """Extract velocity from drone state"""
        vel = drone_state.kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
    
    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return float(np.linalg.norm(pos1 - pos2))
    
    def close(self):
        """Clean up resources"""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("✓ Environment closed")


# Test the environment
if __name__ == "__main__":
    print("Testing Drone Chase Environment...")
    env = DroneChaseEnv()
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    # Test random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Distance: {info['distance_to_target']:.2f}m")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    env.close()