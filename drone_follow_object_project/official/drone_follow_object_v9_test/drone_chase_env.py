"""
Drone Sphere Chasing Environment for AirSim
Continuous action space for smooth drone control
BALANCED AGGRESSIVE VERSION
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
        - vx: velocity in x direction [-15, 15] m/s
        - vy: velocity in y direction [-15, 15] m/s
        - vz: velocity in z direction [-15, 15] m/s
    
    Observation Space:
        - Relative position to target (3D): dx, dy, dz
        - Current velocity (3D): vx, vy, vz
        - Distance to target (1D): euclidean distance
        - Altitude difference (1D): how far drone is from target's altitude
    
    Reward Structure:
        + MASSIVE reward: Hit the target
        + Large reward: Reduce distance to target (aggressive chasing)
        - HUGE Penalty: Increase distance to target (no running away!)
        - HUGE Penalty: Large altitude difference from target
        - CRUSHING Penalty: Going beyond altitude limits
    """
    
    def __init__(self, 
                 max_altitude=0,           
                 min_altitude=-40,         
                 hit_threshold=2.0,
                 max_distance=50,          
                 spawn_radius=25):         
        super(DroneChaseEnv, self).__init__()
        
        # Environment parameters
        self.max_altitude = max_altitude
        self.min_altitude = min_altitude
        self.hit_threshold = hit_threshold
        self.max_distance = max_distance
        self.spawn_radius = spawn_radius
        self.max_altitude_diff = 20.0  # INCREASED from 15! Give more room
        
        # Action space: continuous velocities [vx, vy, vz]
        self.action_space = spaces.Box(
            low=np.array([-15.0, -15.0, -15.0], dtype=np.float32),
            high=np.array([15.0, 15.0, 15.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: [dx, dy, dz, vx, vy, vz, distance, altitude_diff]
        self.observation_space = spaces.Box(
            low=np.array([-200, -200, -200, -15, -15, -15, 0, -100], dtype=np.float32),
            high=np.array([200, 200, 200, 15, 15, 15, 200, 100], dtype=np.float32),
            dtype=np.float32
        )
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # Tracking variables
        self.previous_distance = None
        self.target_position = None
        self.episode_steps = 0
        self.max_episode_steps = 200  # INCREASED from 100! Give drone time to chase
        self.steps_without_progress = 0
        self.max_steps_without_progress = 10  # INCREASED from 2! More forgiving
        self.previous_position = None
        
        # Hit tracking
        self.total_hits = 0
        self.episode_hits = 0
        self.hit_times = []
        self.episode_start_time = None
        
        print("✓ Drone Chase Environment Initialized (BALANCED AGGRESSIVE)")
        print(f"  - Hit threshold: {hit_threshold}m")
        print(f"  - Altitude limits: [{min_altitude}, {max_altitude}]")
        print(f"  - Max distance from origin: {max_distance}m")
        print(f"  - Max altitude diff from target: {self.max_altitude_diff}m")
        print(f"  - Max episode steps: {self.max_episode_steps}")
        print(f"  - Inactivity threshold: {self.max_steps_without_progress} steps")
        print(f"  - Action speed limits: ±15 m/s (AGGRESSIVE)")
    
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
        
        # Clear ALL visual markers (spheres AND text)
        self.client.simFlushPersistentMarkers()
        time.sleep(0.15)
        
        # Spawn target sphere at random location
        self._spawn_target()
        
        # Reset tracking variables
        self.episode_steps = 0
        self.steps_without_progress = 0
        drone_state = self.client.getMultirotorState()
        drone_pos = self._get_position(drone_state)
        self.previous_distance = self._calculate_distance(drone_pos, self.target_position)
        self.previous_position = drone_pos.copy()
        
        # Reset hit tracking
        self.episode_hits = 0
        self.episode_start_time = time.time()
        
        # Get initial observation
        obs = self._get_observation()
        info = {"episode_steps": self.episode_steps}
        
        return obs, info
    
    def step(self, action):
        """Execute one time step within the environment"""
        self.episode_steps += 1
        
        # Apply action: move drone with given velocities
        vx, vy, vz = action
        duration = 1.0  # BACK to 1.0s for smoother control
        
        # Calculate action magnitude
        action_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # Calculate Yaw to face taret
        drone_state = self.client.getMultirotorState()
        drone_pos = self._get_position(drone_state)
        
        # Calculate direction to target
        dx = self.target_position[0] - drone_pos[0]
        dy = self.target_position[1] - drone_pos[1]
    
        # Calculate yaw angle (in degrees)
        target_yaw = math.degrees(math.atan2(dy, dx))
        
        # Use WORLD FRAME velocity control
        self.client.moveByVelocityAsync(
            float(vx), float(vy), float(vz), 
            duration=duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, target_yaw)
        ).join()
        
        # Get current state
        drone_state = self.client.getMultirotorState()
        drone_pos = self._get_position(drone_state)
        
        # Calculate metrics
        current_distance = self._calculate_distance(drone_pos, self.target_position)
        altitude_diff = abs(drone_pos[2] - self.target_position[2])
        
        # REASONABLE Inactivity detection - must move at least 0.5m
        if self.previous_position is not None:
            drone_movement = self._calculate_distance(drone_pos, self.previous_position)
            
            if drone_movement < 0.5:  # REDUCED from 1.0! More reasonable
                self.steps_without_progress += 1
            else:
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
        
        altitude_too_different = altitude_diff > self.max_altitude_diff
        
        # Calculate reward
        reward = self._calculate_reward(
            current_distance=current_distance,
            previous_distance=self.previous_distance,
            altitude_diff=altitude_diff,
            hit_target=hit_target,
            out_of_bounds=out_of_bounds,
            drone_pos=drone_pos,
            action_magnitude=action_magnitude
        )
        
        # Penalties for bad behavior (but not crushing)
        if is_inactive:
            reward -= 50.0  # REDUCED - give drone a chance
            print(f"⚠️  INACTIVE DRONE! No significant movement for {self.max_steps_without_progress} steps")
        
        if altitude_too_different:
            reward -= 30.0  # REDUCED - more forgiving
            print(f"⚠️  ALTITUDE TOO DIFFERENT! Diff={altitude_diff:.1f}m (max={self.max_altitude_diff}m)")
        
        # Update tracking
        self.previous_distance = current_distance
        
        # Termination conditions
        terminated = 0 #hit_target
        
        truncated = (
            out_of_bounds or
            altitude_too_different or
            is_inactive or
            self.episode_steps >= self.max_episode_steps
        )
        
        # DEBUG PRINTS
        if self.episode_steps % 10 == 0 or truncated or terminated:
            avg_time = np.mean(self.hit_times[-10:]) if len(self.hit_times) >= 10 else (np.mean(self.hit_times) if self.hit_times else 0)
            
            print(f"\n[Step {self.episode_steps}/{self.max_episode_steps}] " + 
                  f"Dist={current_distance:.1f}m | " +
                  f"AltDiff={altitude_diff:.1f}m | " +
                  f"Reward={reward:.1f}")
            print(f"  Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}] mag={action_magnitude:.2f}")
            print(f"  Drone: ({drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f})")
            print(f"  Target: ({self.target_position[0]:.1f}, {self.target_position[1]:.1f}, {self.target_position[2]:.1f})")
            print(f"  📊 Hits: Total={self.total_hits} | Episode={self.episode_hits} | AvgTime(last10)={avg_time:.1f}s")
            print(f"  🔄 No progress steps: {self.steps_without_progress}/{self.max_steps_without_progress}")
            
            if truncated:
                print(f"  ❌ TRUNCATED: ", end="")
                if out_of_bounds:
                    print("OUT_OF_BOUNDS", end=" ")
                if altitude_too_different:
                    print("ALTITUDE_TOO_DIFFERENT", end=" ")
                if is_inactive:
                    print("INACTIVE", end=" ")
                if self.episode_steps >= self.max_episode_steps:
                    print("MAX_STEPS", end=" ")
                print()
            
            if terminated:
                print(f"  ✅ TERMINATED: HIT_TARGET!")
        
        # If hit target, record and respawn
        if hit_target:
            self.total_hits += 1
            self.episode_hits += 1
            hit_time = time.time() - self.episode_start_time
            self.hit_times.append(hit_time)
            
            print(f"\n🎯 Target HIT at step {self.episode_steps}! Time: {hit_time:.1f}s")
            
            # CLEAR all markers and wait for clearing to complete
            self.client.simFlushPersistentMarkers()
            time.sleep(0.15)
            
            self._spawn_target()
            self.previous_distance = self._calculate_distance(drone_pos, self.target_position)
            self.steps_without_progress = 0
            self.previous_position = drone_pos.copy()
            
            # Reset timer for next target
            self.episode_start_time = time.time()
        
        # Get observation
        obs = self._get_observation()
        
        info = {
            "episode_steps": self.episode_steps,
            "distance_to_target": current_distance,
            "hit_target": hit_target,
            "out_of_bounds": out_of_bounds,
            "altitude_diff": altitude_diff,
            "altitude_too_different": altitude_too_different,
            "is_inactive": is_inactive,
            "steps_without_progress": self.steps_without_progress
        }
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self, current_distance, previous_distance, 
                          altitude_diff, hit_target, out_of_bounds, drone_pos,
                          action_magnitude):
        """
        BALANCED AGGRESSIVE reward function
        Encourages speed but gives drone time to learn
        """
        reward = 0.0
        reward_breakdown = []
        
        # ═══════════════════════════════════════════════════════════
        # 1. HIT TARGET - MASSIVE JACKPOT!
        # ═══════════════════════════════════════════════════════════
        if hit_target:
            reward += 2000.0  # Big reward!
            reward_breakdown.append(f"HIT_TARGET:+2000.0")
            print(f"  💰 Reward breakdown: {' | '.join(reward_breakdown)} = {reward:.1f}")
            return reward
        
        # ═══════════════════════════════════════════════════════════
        # 2. OUT OF BOUNDS - BIG PENALTY!
        # ═══════════════════════════════════════════════════════════
        if out_of_bounds:
            reward -= 500.0
            reward_breakdown.append(f"OUT_OF_BOUNDS:-500.0")
            print(f"  💰 Reward breakdown: {' | '.join(reward_breakdown)} = {reward:.1f}")
            return reward
        
        # ═══════════════════════════════════════════════════════════
        # 3. DISTANCE CHANGE - Most important signal!
        # ═══════════════════════════════════════════════════════════
        distance_delta = previous_distance - current_distance
        
        if distance_delta > 0.5:  # Getting closer
            distance_reward = distance_delta * 100.0  # Strong reward
            reward += distance_reward
            reward_breakdown.append(f"CLOSER:+{distance_reward:.1f}")
        
        elif distance_delta < -0.5:  # Moving away
            distance_penalty = abs(distance_delta) * 150.0  # Moderate penalty
            reward -= distance_penalty
            reward_breakdown.append(f"AWAY:-{distance_penalty:.1f}")
        
        else:  # Small change or no change
            hover_penalty = 50.0  # Small penalty
            reward -= hover_penalty
            reward_breakdown.append(f"NO_PROGRESS:-{hover_penalty:.1f}")
        
        # ═══════════════════════════════════════════════════════════
        # 4. ACTION MAGNITUDE - Reward for speed
        # ═══════════════════════════════════════════════════════════
        if action_magnitude > 8.0:  # Very fast
            action_reward = action_magnitude * 15.0
            reward += action_reward
            reward_breakdown.append(f"VERY_FAST:+{action_reward:.1f}")
        elif action_magnitude > 5.0:  # Fast
            action_reward = action_magnitude * 10.0
            reward += action_reward
            reward_breakdown.append(f"FAST:+{action_reward:.1f}")
        elif action_magnitude > 3.0:  # Decent speed
            action_reward = action_magnitude * 5.0
            reward += action_reward
            reward_breakdown.append(f"MOVING:+{action_reward:.1f}")
        elif action_magnitude < 1.0:  # Too slow
            action_penalty = 30.0
            reward -= action_penalty
            reward_breakdown.append(f"TOO_SLOW:-{action_penalty:.1f}")
        
        # ═══════════════════════════════════════════════════════════
        # 5. DIRECTION ALIGNMENT - Bonus for moving toward target
        # ═══════════════════════════════════════════════════════════
        if hasattr(self, 'target_position') and current_distance > 0:
            dx_target = self.target_position[0] - drone_pos[0]
            dy_target = self.target_position[1] - drone_pos[1]
            dz_target = self.target_position[2] - drone_pos[2]
            
            target_dir_mag = np.sqrt(dx_target**2 + dy_target**2 + dz_target**2)
            if target_dir_mag > 0:
                # Closing distance AND moving fast = good!
                if distance_delta > 0 and action_magnitude > 3.0:
                    alignment_reward = 30.0
                    reward += alignment_reward
                    reward_breakdown.append(f"ALIGNED:+{alignment_reward:.1f}")
        
        # ═══════════════════════════════════════════════════════════
        # 6. ALTITUDE MATCHING
        # ═══════════════════════════════════════════════════════════
        if altitude_diff > 5.0:
            altitude_penalty = altitude_diff * 3.0  # Moderate penalty
            reward -= altitude_penalty
            reward_breakdown.append(f"ALTITUDE:-{altitude_penalty:.1f}")
        
        # ═══════════════════════════════════════════════════════════
        # 7. DISTANCE PENALTY - Being far is bad
        # ═══════════════════════════════════════════════════════════
        far_penalty = current_distance * 0.5
        reward -= far_penalty
        reward_breakdown.append(f"FAR:-{far_penalty:.1f}")
        
        # ═══════════════════════════════════════════════════════════
        # 8. TIME PENALTY - Encourages finishing
        # ═══════════════════════════════════════════════════════════
        time_penalty = 2.0
        reward -= time_penalty
        reward_breakdown.append(f"TIME:-{time_penalty:.1f}")
        
        # Print breakdown every 10 steps
        if hasattr(self, 'episode_steps') and self.episode_steps % 10 == 0:
            print(f"  💰 Reward breakdown: {' | '.join(reward_breakdown)} = {reward:.1f}")
        
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
        altitude_diff = dz
        
        obs = np.array([dx, dy, dz, vx, vy, vz, distance, altitude_diff], 
                       dtype=np.float32)
        
        return obs
    
    def _create_point_cloud_sphere(self, position, radius=2.0, color=[1.0, 0.0, 0.0, 1.0]):
        """Create a visual sphere made of points in AirSim"""
        points = []
        num_points = 150
        
        for i in range(num_points):
            phi = np.pi * (3. - np.sqrt(5.))
            y_offset = 1 - (i / float(num_points - 1)) * 2
            radius_at_y = np.sqrt(1 - y_offset * y_offset)
            theta = phi * i
            
            x = np.cos(theta) * radius_at_y * radius
            y = np.sin(theta) * radius_at_y * radius
            z = y_offset * radius
            
            point = airsim.Vector3r(
                position[0] + x,
                position[1] + y,
                position[2] + z
            )
            points.append(point)
        
        self.client.simPlotPoints(
            points,
            color_rgba=color,
            size=15.0,
            duration=120.0,
            is_persistent=True
        )
    
    def _spawn_target(self):
        """Spawn target sphere at random location with visual representation and coordinates text"""
        # FIRST: Clear old markers explicitly
        self.client.simFlushPersistentMarkers()
        time.sleep(0.15)
        
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(10, self.spawn_radius)
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = np.random.uniform(-30, -10)
        
        self.target_position = np.array([x, y, z], dtype=np.float32)
        
        print(f"🎯 Target spawned at: ({x:.1f}, {y:.1f}, {z:.1f})")
        
        # Create visual sphere
        self._create_point_cloud_sphere(
            position=[x, y, z],
            radius=2.0,
            color=[1.0, 0.0, 0.0, 1.0]
        )
        
        # Display X/Y/Z coordinates as text below the sphere
        self.client.simPlotStrings(
            strings=["X: {:.1f}\nY: {:.1f}\nZ: {:.1f}".format(x, y, z)],
            positions=[airsim.Vector3r(x, y, z - 3)],
            scale=2.0,
            color_rgba=[1.0, 1.0, 1.0, 1.0],
            duration=120.0
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
    print("Testing BALANCED AGGRESSIVE Drone Chase Environment...")
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