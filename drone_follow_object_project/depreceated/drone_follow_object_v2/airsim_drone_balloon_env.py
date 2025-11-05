import gymnasium as gym
import numpy as np
import airsim
import time
from gymnasium import spaces


class AirSimDroneBalloonEnv(gym.Env):
    """
    AirSim environment where a drone chases and hits spherical targets in 3D space.
    
    Observation Space: [drone_x, drone_y, drone_z, target_x, target_y, target_z, 
                        drone_vx, drone_vy, drone_vz]
    Action Space: Discrete(7) - [stay, forward, backward, left, right, up, down]
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 20}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Environment parameters
        self.drone_speed = 5.0  # m/s
        self.max_score = 6
        self.hit_distance = 3.0  # meters - distance threshold for hitting target
        
        # Flight boundaries (in meters, NED coordinates)
        # NED = North-East-Down (AirSim coordinate system)
        # For drone movement: X=left/right, Y=forward/backward, Z=down/up
        # IMPORTANT: Keep boundaries small to fit within AirSim world
        self.boundary_min = np.array([-20.0, -20.0, -30.0])  # min x, y, z (smaller area!)
        self.boundary_max = np.array([20.0, 20.0, -5.0])     # max x, y, z
        # Note: Z is negative because in NED, negative Z = up in the air
        
        # Spawn radius - keep drone and targets close together
        self.spawn_radius = 15.0  # Maximum distance from center (0,0)
        
        # Observation space: [drone_xyz, target_xyz, drone_velocity_xyz]
        # Normalized to roughly [-1, 1] range
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )
        
        # Action space: 0=stay, 1=forward, 2=backward, 3=left, 4=right, 5=up, 6=down
        self.action_space = spaces.Discrete(7)
        
        # Rendering
        self.render_mode = render_mode
        
        # State variables
        self.drone_pos = None
        self.target_pos = None
        self.score = 0
        self.steps = 0
        self.max_steps_per_episode = 1000
        
        # AirSim client
        self.client = None
        self._connect_airsim()
        
    def _connect_airsim(self):
        """Connect to AirSim simulator"""
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            print("✓ Connected to AirSim successfully!")
        except Exception as e:
            print(f"✗ Failed to connect to AirSim: {e}")
            print("Make sure AirSim is running before starting training!")
            raise
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset score and steps
        self.score = 0
        self.steps = 0
        
        # Reset drone to starting position
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Take off to initial height
        print("Taking off...")
        self.client.takeoffAsync().join()
        time.sleep(1)  # Wait for stabilization
        
        # Move to a random starting position (CLOSER to center, within spawn radius)
        start_x = self.np_random.uniform(-10, 10)  # Reduced from -20,20
        start_y = self.np_random.uniform(-10, 10)  # Reduced from -20,20
        start_z = -10.0  # 10 meters up (fixed altitude)
        
        self.client.moveToPositionAsync(
            start_x, start_y, start_z, 
            velocity=3.0
        ).join()
        
        # Get actual drone position
        self.drone_pos = self._get_drone_position()
        
        # Spawn target sphere (within reasonable distance)
        self.target_pos = self._spawn_target()
        self._create_target_visual()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self.steps += 1
        
        # Get current drone state
        previous_pos = self.drone_pos.copy()
        
        # Calculate velocity vector based on action
        velocity = np.zeros(3)
        
        # FIXED: In AirSim, forward/backward is Y-axis, left/right is X-axis
        if action == 1:  # forward (positive Y)
            velocity[1] = self.drone_speed
        elif action == 2:  # backward (negative Y)
            velocity[1] = -self.drone_speed
        elif action == 3:  # left (negative X)
            velocity[0] = -self.drone_speed
        elif action == 4:  # right (positive X)
            velocity[0] = self.drone_speed
        elif action == 5:  # up (negative Z in NED)
            velocity[2] = -self.drone_speed
        elif action == 6:  # down (positive Z in NED)
            velocity[2] = self.drone_speed
        # action == 0: stay (hover in place)
        
        # Execute movement in AirSim
        if action != 0:
            # Move with velocity for a short duration
            duration = 0.5  # seconds
            self.client.moveByVelocityAsync(
                velocity[0], velocity[1], velocity[2],
                duration=duration
            ).join()
        else:
            # Hover in place
            self.client.hoverAsync().join()
            time.sleep(0.1)
        
        # Get new drone position
        self.drone_pos = self._get_drone_position()
        
        # Check if drone is out of bounds
        out_of_bounds = np.any(self.drone_pos < self.boundary_min) or \
                        np.any(self.drone_pos > self.boundary_max)
        
        if out_of_bounds:
            # Penalize and revert position (same as 2D code)
            reward = -1.0
            self.drone_pos = previous_pos
        else:
            # ====== CHASE LOGIC ======
            # Calculate Euclidean distance between drone center (X,Y,Z) and sphere center (X,Y,Z)
            drone_center = self.drone_pos  # Drone position [x, y, z]
            sphere_center = self.target_pos  # Sphere position [x, y, z]
            
            # Euclidean distance in 3D space
            distance = np.linalg.norm(drone_center - sphere_center)
            
            # Check if drone hit the sphere (within hit_distance threshold)
            if distance < self.hit_distance:
                # SUCCESS! Drone reached the sphere
                reward = 10.0  # Big reward for hitting target
                self.score += 1
                
                print(f"🎯 Hit! Score: {self.score}/{self.max_score}, Distance: {distance:.2f}m")
                
                # Spawn new sphere if not reached max score
                if self.score < self.max_score:
                    self.target_pos = self._spawn_target()
                    self._create_target_visual()
            else:
                # ENCOURAGE CHASING: Reward for getting closer to sphere
                prev_distance = np.linalg.norm(previous_pos - sphere_center)
                distance_change = prev_distance - distance
                
                # Reward proportional to how much closer we got
                # Positive if moved closer, negative if moved away
                chase_reward = distance_change * 0.01
                
                # Small time penalty to encourage efficiency
                time_penalty = -0.01
                
                # Total reward = chase behavior + time penalty
                reward = chase_reward + time_penalty
                
                # Debug: Show if drone is getting closer or farther
                if distance_change > 0:
                    status = "→ closer"
                elif distance_change < 0:
                    status = "← farther"
                else:
                    status = "= same"
                
                if self.steps % 50 == 0:  # Print every 50 steps
                    print(f"Distance: {distance:.2f}m {status}, Reward: {reward:.4f}")
        
        # Check if episode is done
        terminated = (self.score >= self.max_score)
        truncated = (self.steps >= self.max_steps_per_episode)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_drone_position(self):
        """Get current drone position in NED coordinates"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
    
    def _get_drone_velocity(self):
        """Get current drone velocity"""
        state = self.client.getMultirotorState()
        vel = state.kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
    
    def _spawn_target(self):
        """Spawn target at random position away from drone but within world bounds"""
        max_attempts = 100
        for attempt in range(max_attempts):
            # Random position within smaller boundaries (fits AirSim world)
            pos = np.array([
                self.np_random.uniform(-15.0, 15.0),  # X: within 15m of center
                self.np_random.uniform(-15.0, 15.0),  # Y: within 15m of center
                self.np_random.uniform(-25.0, -8.0)   # Z: 8-25m altitude
            ], dtype=np.float32)
            
            # Calculate distance from drone to target
            distance = np.linalg.norm(pos - self.drone_pos)
            
            # Ensure target spawns between 8-20 meters away (not too close, not too far)
            if 8.0 < distance < 20.0:
                return pos
        
        # Fallback: if can't find good position, spawn at fixed distance
        angle = self.np_random.uniform(0, 2 * np.pi)
        distance = 12.0  # Fixed distance
        return np.array([
            self.drone_pos[0] + distance * np.cos(angle),
            self.drone_pos[1] + distance * np.sin(angle),
            self.np_random.uniform(-20.0, -10.0)
        ], dtype=np.float32)
    
    def _create_target_visual(self):
        """Create a visual sphere in AirSim using point cloud method"""
        # Clear any previous markers
        try:
            self.client.simFlushPersistentMarkers()
        except:
            pass
        
        # Create a sphere made of points (works in all AirSim versions!)
        points = []
        radius = 2.0  # 2 meter radius sphere
        num_points = 150  # Number of points to create sphere
        
        # Generate points on sphere surface using Fibonacci sphere algorithm
        for i in range(num_points):
            phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
            y_offset = 1 - (i / float(num_points - 1)) * 2  # y from -1 to 1
            radius_at_y = np.sqrt(1 - y_offset * y_offset)
            
            theta = phi * i
            
            x = float(np.cos(theta) * radius_at_y * radius)  # Convert to Python float
            y = float(np.sin(theta) * radius_at_y * radius)
            z = float(y_offset * radius)
            
            # CRITICAL FIX: Convert numpy floats to Python floats
            point = airsim.Vector3r(
                float(self.target_pos[0]) + x,
                float(self.target_pos[1]) + y,
                float(self.target_pos[2]) + z
            )
            points.append(point)
        
        # Draw the sphere
        try:
            self.client.simPlotPoints(
                points,
                color_rgba=[1.0, 0.0, 0.0, 1.0],  # Red color
                size=15.0,  # Point size in pixels
                duration=300.0,  # How long it stays (seconds)
                is_persistent=True
            )
            
            if self.render_mode == "human":
                print(f"✓ Target spawned at: X={self.target_pos[0]:.1f}, "
                      f"Y={self.target_pos[1]:.1f}, Z={self.target_pos[2]:.1f}")
        except Exception as e:
            # Fallback to line marker if points fail
            if self.render_mode == "human":
                print(f"⚠️ Point cloud failed: {e}")
                print(f"  Trying line marker fallback...")
            
            try:
                # Fallback: Create line marker instead
                self._create_line_marker()
            except Exception as e2:
                if self.render_mode == "human":
                    print(f"⚠️ Line marker also failed: {e2}")
                    print(f"  Target at: X={self.target_pos[0]:.1f}, "
                          f"Y={self.target_pos[1]:.1f}, Z={self.target_pos[2]:.1f}")
    
    def _create_line_marker(self):
        """Fallback method: Create an X marker using lines"""
        marker_size = 2.0
        x, y, z = float(self.target_pos[0]), float(self.target_pos[1]), float(self.target_pos[2])
        
        lines = [
            airsim.Vector3r(x - marker_size, y - marker_size, z),
            airsim.Vector3r(x + marker_size, y + marker_size, z),
            airsim.Vector3r(x + marker_size, y - marker_size, z),
            airsim.Vector3r(x - marker_size, y + marker_size, z),
            airsim.Vector3r(x, y, z - marker_size),
            airsim.Vector3r(x, y, z + marker_size)
        ]
        
        self.client.simPlotLineList(
            lines,
            color_rgba=[1.0, 0.0, 0.0, 1.0],
            thickness=0.3,
            duration=300.0,
            is_persistent=True
        )
        
        if self.render_mode == "human":
            print(f"✓ Line marker created at: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
    
    def _get_observation(self):
        """Get current observation (normalized positions and velocity)"""
        # Normalize positions to roughly [-1, 1]
        norm_drone_pos = self.drone_pos / 50.0
        norm_target_pos = self.target_pos / 50.0
        
        # Get and normalize velocity
        drone_vel = self._get_drone_velocity()
        norm_velocity = drone_vel / 10.0  # Normalize by max expected velocity
        
        return np.concatenate([
            norm_drone_pos,
            norm_target_pos,
            norm_velocity
        ]).astype(np.float32)
    
    def _get_info(self):
        """Get additional info"""
        distance = np.linalg.norm(self.drone_pos - self.target_pos)
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_to_target": distance,
            "drone_position": self.drone_pos.copy(),
            "target_position": self.target_pos.copy()
        }
    
    def render(self):
        """Render is handled by AirSim's visualization"""
        if self.render_mode == "human":
            # AirSim provides its own rendering
            # You can optionally grab camera images here
            pass
    
    def close(self):
        """Clean up AirSim connection"""
        if self.client:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("AirSim environment closed")


# Test the environment
if __name__ == "__main__":
    print("Testing AirSim Drone Balloon Environment")
    print("=" * 50)
    print("Make sure AirSim is running!")
    print()
    
    env = AirSimDroneBalloonEnv(render_mode="human")
    obs, info = env.reset()
    
    print("\nTesting environment with random actions...")
    print("Watch the drone in AirSim!")
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}: Action={action}, Reward={reward:.2f}, "
              f"Distance={info['distance_to_target']:.2f}m, Score={info['score']}")
        
        if terminated or truncated:
            print(f"\nEpisode finished! Final Score: {info['score']}")
            obs, info = env.reset()
    
    env.close()