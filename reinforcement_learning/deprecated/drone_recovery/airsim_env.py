import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

class AirSimDroneEnv(gym.Env):
    """
    Custom gym environment with 3-stage curriculum.
    
    FIXES APPLIED:
    1. Fixed crash detection: z > -0.5 instead of z > -1
    2. Added debug logging for first 5 steps
    3. Fixed episode length tracking
    4. Added proper error handling
    """
    
    def __init__(self, stage=1):
        super().__init__()
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # Load normalization parameters
        try:
            norm_params = np.load('data/normalization_params.npy', allow_pickle=True).item()
            self.action_mean = norm_params['action_mean']
            self.action_std = norm_params['action_std']
            print("✅ Loaded action normalization parameters")
        except:
            print("⚠️  Warning: Could not load normalization params, using defaults")
            self.action_mean = np.zeros(4)
            self.action_std = np.ones(4)
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(13,), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-3.0, 
            high=3.0, 
            shape=(4,), 
            dtype=np.float32
        )
        
        self.max_steps = 500
        self.current_step = 0
        
        # Curriculum stage
        self.stage = stage
        self.disturbance_probability = self._get_disturbance_probability()
        
        # Determine rotor format
        self.rotor_format = self._detect_rotor_format()
        
        print(f"🎮 Environment initialized: Stage {self.stage}")
        print(f"   Disturbance probability: {self.disturbance_probability}")
        
    def _detect_rotor_format(self):
        """Detect if rotors return dict or object"""
        test_rotor = self.client.getRotorStates()
        if hasattr(test_rotor, 'rotors') and len(test_rotor.rotors) > 0:
            if isinstance(test_rotor.rotors[0], dict):
                return 'dict'
            else:
                return 'object'
        return 'unknown'
    
    def _get_motor_speeds(self):
        """Get motor speeds regardless of format"""
        rotor_states = self.client.getRotorStates()
        if not hasattr(rotor_states, 'rotors') or len(rotor_states.rotors) == 0:
            return None
        
        if self.rotor_format == 'dict':
            return np.array([r['speed'] for r in rotor_states.rotors])
        else:
            return np.array([r.speed for r in rotor_states.rotors])
    
    def _get_disturbance_probability(self):
        """How often to inject disturbances"""
        probabilities = {
            1: 0.02,
            2: 0.10,
            3: 0.15
        }
        return probabilities.get(self.stage, 0.02)
    
    def reset(self, seed=None, options=None):
        """
        Reset drone to starting position.
        
        FIXES:
        - Added debug logging
        - Added error handling
        - Increased stabilization time
        """
        super().reset(seed=seed)
        
        try:
            # Reset AirSim
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            
            # Takeoff
            self.client.takeoffAsync().join()
            time.sleep(2)  # CHANGED: Increased from 1 to 2 seconds
            
            # Move to hover position and WAIT
            self.client.moveToPositionAsync(0, 0, -10, 1).join()
            time.sleep(2)  # CHANGED: Increased from 1 to 2 seconds
            
            self.current_step = 0
            
            obs = self._get_observation()
            
            # Debug: Print initial state for first episode
            if not hasattr(self, '_reset_count'):
                self._reset_count = 0
            
            self._reset_count += 1
            
            if self._reset_count <= 3:
                print(f"\n  🔄 [RESET #{self._reset_count}] Initial state:")
                print(f"      Position: ({obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f})")
                print(f"      Orientation: qw={obs[3]:.3f}, qx={obs[4]:.3f}, qy={obs[5]:.3f}, qz={obs[6]:.3f}")
                print(f"      Linear velocity: ({obs[7]:.2f}, {obs[8]:.2f}, {obs[9]:.2f})")
                print(f"      Angular velocity: ({obs[10]:.2f}, {obs[11]:.2f}, {obs[12]:.2f})")
            
            info = {}
            
            return obs, info
            
        except Exception as e:
            print(f"  ❌ [RESET] Error during reset: {e}")
            raise
    
    def _get_observation(self):
        """Get current drone state"""
        state = self.client.getMultirotorState()
        
        obs = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val,
            state.kinematics_estimated.orientation.w_val,
            state.kinematics_estimated.orientation.x_val,
            state.kinematics_estimated.orientation.y_val,
            state.kinematics_estimated.orientation.z_val,
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val,
            state.kinematics_estimated.angular_velocity.x_val,
            state.kinematics_estimated.angular_velocity.y_val,
            state.kinematics_estimated.angular_velocity.z_val,
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self, obs):
        """
        Reward function that adapts to curriculum stage.
        
        Components:
        - Position error (closer to target = better)
        - Orientation reward (upright = better)
        - Velocity penalties (stable = better)
        """
        pos_x, pos_y, pos_z = obs[0], obs[1], obs[2]
        qw, qx, qy, qz = obs[3], obs[4], obs[5], obs[6]
        lin_vel = obs[7:10]
        ang_vel = obs[10:13]
        
        # Target: hover at (0, 0, -10)
        target = np.array([0, 0, -10])
        position = np.array([pos_x, pos_y, pos_z])
        
        # Position error
        pos_error = np.linalg.norm(position - target)
        pos_reward = -pos_error
        
        # Orientation reward (upright = qw close to 1)
        orientation_reward = 10 * qw
        
        # Velocity penalties
        vel_penalty = -0.1 * np.linalg.norm(lin_vel)
        ang_vel_penalty = -0.5 * np.linalg.norm(ang_vel)
        
        # Stage-specific bonuses
        if self.stage >= 2:
            stability_bonus = 5.0 if abs(qw) > 0.9 else 0.0
            orientation_reward += stability_bonus
        
        if self.stage == 3:
            ang_vel_norm = np.linalg.norm(ang_vel)
            recovery_bonus = 10.0 * np.exp(-ang_vel_norm)
            orientation_reward += recovery_bonus
        
        # Total reward
        reward = pos_reward + orientation_reward + vel_penalty + ang_vel_penalty
        
        return reward
    
    def step(self, action):
        """
        Execute action and return next state.
        
        FIXED: Proper action denormalization and PWM conversion
        """
        
        # Debug: print first few steps
        debug = (self.current_step < 5 and self._reset_count <= 2)
        
        if debug:
            print(f"    [STEP {self.current_step}] Raw action from PPO: {action}")
        
        try:
            # FIXED: Denormalize action back to motor_thrusts (speed^2)
            action_denorm = action * self.action_std + self.action_mean
            
            if debug:
                print(f"    [STEP {self.current_step}] Denormalized (thrust): {action_denorm}")
            
            # FIXED: Convert thrust back to motor speeds
            # During collection: thrust = speed^2
            # So: speed = sqrt(thrust)
            # Handle negative thrusts (set to minimum)
            motor_speeds = np.sqrt(np.maximum(action_denorm, 0))
            
            if debug:
                print(f"    [STEP {self.current_step}] Motor speeds (RPM): {motor_speeds}")
            
            # FIXED: Normalize motor speeds to PWM [0, 1]
            # Typical hover speed is around 4000-5000 RPM
            # Max speed is around 8000-10000 RPM
            # So PWM = speed / max_speed
            MAX_MOTOR_SPEED = 6000  # Adjust based on your drone
            motor_pwms = motor_speeds / MAX_MOTOR_SPEED
            motor_pwms = np.clip(motor_pwms, 0.0, 1.0)
            
            # SAFETY: If all motors too low, set to minimum hover thrust
            if np.all(motor_pwms < 0.3):
                if debug:
                    print(f"    ⚠️  [STEP {self.current_step}] WARNING: All motors < 0.3, setting to 0.5")
                motor_pwms = np.full(4, 0.5)
            
            if debug:
                print(f"    [STEP {self.current_step}] Motor PWMs: {motor_pwms}")
                print(f"    [STEP {self.current_step}] PWM sum: {motor_pwms.sum():.2f} (should be ~2.0-2.4 for hover)")
            
            # Apply motor commands
            self.client.moveByMotorPWMsAsync(
                float(motor_pwms[0]),
                float(motor_pwms[1]),
                float(motor_pwms[2]),
                float(motor_pwms[3]),
                duration=0.1
            ).join()
            
            # Randomly inject disturbances
            if np.random.random() < self.disturbance_probability:
                if debug:
                    print(f"    [STEP {self.current_step}] 💥 Injecting disturbance")
                self._inject_stage_appropriate_disturbance()
            
            # Get new observation
            obs = self._get_observation()
            
            if debug:
                print(f"    [STEP {self.current_step}] Position: ({obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f}), qw={obs[3]:.3f}")
            
            # Compute reward
            reward = self._compute_reward(obs)
            
            # Check termination conditions
            self.current_step += 1
            terminated = False
            truncated = False
            
            # Crash detection
            if obs[2] > -0.5:
                terminated = True
                reward = -100
                if debug or self.current_step <= 10:
                    print(f"    ❌ [STEP {self.current_step}] CRASHED: z={obs[2]:.2f}")
            
            # Flip detection
            if obs[3] < 0.3 and self.stage >= 2:
                terminated = True
                reward = -50
                if debug or self.current_step <= 10:
                    print(f"    ❌ [STEP {self.current_step}] FLIPPED: qw={obs[3]:.3f}")
            
            # Out of bounds
            horizontal_dist = np.linalg.norm(obs[0:2])
            if horizontal_dist > 20:
                terminated = True
                reward = -50
                if debug or self.current_step <= 10:
                    print(f"    ❌ [STEP {self.current_step}] OUT OF BOUNDS: {horizontal_dist:.2f}m")
            
            # Max steps
            if self.current_step >= self.max_steps:
                truncated = True
            
            if debug:
                print(f"    [STEP {self.current_step}] Reward: {reward:.2f}, Term: {terminated}, Trunc: {truncated}\n")
            
            info = {}
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"    ❌ [STEP {self.current_step}] Error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _inject_stage_appropriate_disturbance(self):
        """
        Inject disturbance based on stage.
        Stage 1: Light position offsets
        Stage 2: Medium position offsets
        Stage 3: Heavy position offsets + orientation changes
        """
        pose = self.client.simGetVehiclePose()
        
        if self.stage == 1:
            # Light: Small position offset
            pose.position.x_val += np.random.uniform(-1, 1)
            pose.position.y_val += np.random.uniform(-1, 1)
            
        elif self.stage == 2:
            # Medium: Larger position offset
            pose.position.x_val += np.random.uniform(-3, 3)
            pose.position.y_val += np.random.uniform(-3, 3)
            
        elif self.stage == 3:
            # Heavy: Large position offset + rotation
            pose.position.x_val += np.random.uniform(-5, 5)
            pose.position.y_val += np.random.uniform(-5, 5)
            
            try:
                from scipy.spatial.transform import Rotation as R
                import math
                
                roll = np.random.uniform(-math.pi/3, math.pi/3)
                pitch = np.random.uniform(-math.pi/3, math.pi/3)
                yaw = 0
                
                rot = R.from_euler('xyz', [roll, pitch, yaw])
                quat = rot.as_quat()
                
                pose.orientation = airsim.Quaternionr(quat[0], quat[1], quat[2], quat[3])
            except ImportError:
                pass
        
        self.client.simSetVehiclePose(pose, True)
    
    def set_stage(self, stage):
        """Change curriculum stage"""
        self.stage = stage
        self.disturbance_probability = self._get_disturbance_probability()
        print(f"🔄 Stage changed to {stage}")