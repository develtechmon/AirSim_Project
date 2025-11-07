"""
PID EXPERT CONTROLLER FOR HOVER
================================
This is the "teacher" that generates perfect hovering demonstrations.
Based on cascaded PID control (industry standard for drones).

The neural network will learn to imitate this expert's behavior.

python pid_expert.py (5 min)

Taking off...
Moving to 10m altitude...
✓ PID Expert Controller Initialized
Target: Hover at 10.0m
Control frequency: 20 Hz

Running PID hover test for 100 steps (5 seconds)...
Watch the drone - it should hover stably!

Step   0: Alt=10.38m, Dist from center=0.00m
Step  20: Alt=10.23m, Dist from center=0.00m
Step  40: Alt=10.22m, Dist from center=0.00m
Step  60: Alt=10.18m, Dist from center=0.00m
Step  80: Alt=10.14m, Dist from center=0.00m

======================================================================
RESULTS
======================================================================
Mean altitude: 10.193m (target: 10.0m)
Std deviation: 0.054m
Max error: 0.381m

    PID Expert is EXCELLENT! Ready to generate demonstrations.
======================================================================

python pid_expert.py (5 min)
"""

import airsim
import numpy as np
import time


class PIDController:
    """Simple PID controller"""
    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.integral = 0
        self.previous_error = 0
    
    def update(self, error, dt):
        """Calculate PID output"""
        # Proportional
        p_term = self.kp * error
        
        # Integral
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative
        d_term = self.kd * (error - self.previous_error) / dt if dt > 0 else 0
        self.previous_error = error
        
        # Total output
        output = p_term + i_term + d_term
        
        # Apply limits
        if self.output_limits:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        return output
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0
        self.previous_error = 0


class PIDExpertHover:
    """
    Expert PID controller for hovering
    Uses cascaded control:
      - Position controller (outer loop) → desired velocity
      - Velocity controller (inner loop) → throttle
    """
    
    def __init__(self, target_altitude=10.0):
        self.target_altitude = target_altitude
        self.dt = 0.05  # 20 Hz control
        
        # Position PIDs (x, y, z)
        self.pid_x = PIDController(kp=0.5, ki=0.01, kd=0.3, output_limits=(-3, 3))
        self.pid_y = PIDController(kp=0.5, ki=0.01, kd=0.3, output_limits=(-3, 3))
        self.pid_z = PIDController(kp=0.8, ki=0.02, kd=0.5, output_limits=(-3, 3))
        
        # Velocity PIDs (converts desired velocity to action)
        self.pid_vx = PIDController(kp=0.3, ki=0.01, kd=0.1, output_limits=(-5, 5))
        self.pid_vy = PIDController(kp=0.3, ki=0.01, kd=0.1, output_limits=(-5, 5))
        self.pid_vz = PIDController(kp=0.4, ki=0.02, kd=0.15, output_limits=(-5, 5))
        
        print("✓ PID Expert Controller Initialized")
        print(f"  Target: Hover at {target_altitude}m")
        print(f"  Control frequency: {1/self.dt:.0f} Hz")
    
    def get_action(self, state):
        """
        Given current state, return expert action
        
        Args:
            state: dict with keys:
                - position: [x, y, z]
                - velocity: [vx, vy, vz]
                - orientation: [qw, qx, qy, qz]
        
        Returns:
            action: [vx_cmd, vy_cmd, vz_cmd] velocity commands
        """
        pos = state['position']
        vel = state['velocity']
        
        # Position errors
        error_x = 0.0 - pos[0]  # Target x=0
        error_y = 0.0 - pos[1]  # Target y=0
        error_z = -self.target_altitude - pos[2]  # AirSim uses NED (z negative)
        
        # Position PIDs output desired velocities
        desired_vx = self.pid_x.update(error_x, self.dt)
        desired_vy = self.pid_y.update(error_y, self.dt)
        desired_vz = self.pid_z.update(error_z, self.dt)
        
        # Velocity errors
        vel_error_x = desired_vx - vel[0]
        vel_error_y = desired_vy - vel[1]
        vel_error_z = desired_vz - vel[2]
        
        # Velocity PIDs output velocity commands
        vx_cmd = self.pid_vx.update(vel_error_x, self.dt)
        vy_cmd = self.pid_vy.update(vel_error_y, self.dt)
        vz_cmd = self.pid_vz.update(vel_error_z, self.dt)
        
        return np.array([vx_cmd, vy_cmd, vz_cmd], dtype=np.float32)
    
    def reset(self):
        """Reset all PIDs"""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_vx.reset()
        self.pid_vy.reset()
        self.pid_vz.reset()


def test_pid_expert():
    """Test the PID expert controller"""
    print("\n" + "="*70)
    print("🧪 TESTING PID EXPERT CONTROLLER")
    print("="*70 + "\n")
    
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    # Takeoff
    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)
    
    # Move to starting position
    print("Moving to 10m altitude...")
    client.moveToPositionAsync(0, 0, -10, 5).join()
    time.sleep(2)
    
    # Create PID expert
    expert = PIDExpertHover(target_altitude=10.0)
    
    print("\n🎯 Running PID hover test for 100 steps (5 seconds)...")
    print("Watch the drone - it should hover stably!\n")
    
    positions = []
    for step in range(100):
        # Get state
        drone_state = client.getMultirotorState()
        pos = drone_state.kinematics_estimated.position
        vel = drone_state.kinematics_estimated.linear_velocity
        ori = drone_state.kinematics_estimated.orientation
        
        state = {
            'position': np.array([pos.x_val, pos.y_val, pos.z_val]),
            'velocity': np.array([vel.x_val, vel.y_val, vel.z_val]),
            'orientation': np.array([ori.w_val, ori.x_val, ori.y_val, ori.z_val])
        }
        
        # Get expert action
        action = expert.get_action(state)
        
        # Execute
        client.moveByVelocityAsync(
            float(action[0]), float(action[1]), float(action[2]),
            duration=expert.dt,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        ).join()
        
        positions.append(-pos.z_val)  # Altitude (positive)
        
        if step % 20 == 0:
            alt = -pos.z_val
            dist = np.linalg.norm([pos.x_val, pos.y_val])
            print(f"Step {step:3d}: Alt={alt:.2f}m, Dist from center={dist:.2f}m")
    
    # Statistics
    positions = np.array(positions)
    mean_alt = np.mean(positions)
    std_alt = np.std(positions)
    max_error = np.max(np.abs(positions - 10.0))
    
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"Mean altitude: {mean_alt:.3f}m (target: 10.0m)")
    print(f"Std deviation: {std_alt:.3f}m")
    print(f"Max error: {max_error:.3f}m")
    
    if std_alt < 0.3 and max_error < 0.5:
        print("\n✅ PID Expert is EXCELLENT! Ready to generate demonstrations.")
    elif std_alt < 0.5:
        print("\n✅ PID Expert is GOOD! Can be used for demonstrations.")
    else:
        print("\n⚠️  PID Expert needs tuning. Adjust PID gains.")
    
    print("="*70 + "\n")
    
    # Land
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)


if __name__ == "__main__":
    test_pid_expert()